"""Base API class for model inference.

Provides common functionality shared by all API backends:
message preprocessing, image encoding, retry logic, and parallel
request processing.
"""

import concurrent.futures
import copy as cp
import ctypes
import gc
import random as rd
import time
from abc import abstractmethod

try:
    _libc = ctypes.CDLL("libc.so.6")
    _malloc_trim = _libc.malloc_trim
except (OSError, AttributeError):

    def _malloc_trim(x):
        return 0


import numpy as np
from PIL import Image
from tqdm import tqdm

from datastudio.utils.json_parser import load_str_to_dict
from datastudio.utils.vision import encode_image_to_base64, parse_file


class BaseAPI:
    """Base API class providing common functionality for all API models.

    This class handles:
    - Message preprocessing and formatting
    - Image encoding (base64)
    - Retry logic with exponential backoff
    - Parallel request processing
    """

    allowed_types = ["text", "image", "image_url", "system"]

    def __init__(
        self,
        retry=10,
        wait=5,
        timeout=(30, 1800),
        logger=None,
        thread_num=384,
        fail_msg="Failed to obtain answer via API.",
        openai_format=True,
        image_key_name="image",
        return_dict=True,
        use_system_proxy=False,
        **kwargs,
    ):
        """Initialize the base API.

        Args:
            retry: Number of retry attempts on API failure.
            wait: Wait time between retries (seconds).
            timeout: Request timeout as (connect_timeout, read_timeout) tuple.
            logger: Logger instance.
            thread_num: Number of parallel threads.
            fail_msg: Message returned on failure.
            openai_format: Whether to use OpenAI message format.
            image_key_name: Key name for image data.
            return_dict: Whether to parse response as dict.
            use_system_proxy: Whether to use system proxy.
            **kwargs: Additional arguments passed to generate_inner.
        """
        self.retry = retry
        self.wait = wait
        self.timeout = timeout
        self.logger = logger
        self.fail_msg = fail_msg
        self.thread_num = thread_num
        self.default_kwargs = kwargs if kwargs else {}
        self.image_key_name = image_key_name
        self.openai_format = openai_format
        self.return_dict = return_dict
        self.use_system_proxy = use_system_proxy

    @staticmethod
    def _collect_memory(trim: bool = False):
        """Run garbage collection and optionally trim glibc malloc arenas.

        Args:
            trim: If True, call malloc_trim(0) to return freed memory to the OS.
        """
        gc.collect()
        if trim:
            _malloc_trim(0)

    @staticmethod
    def _clear_message_buffers(messages):
        """Clear all message buffers to release memory.

        Args:
            messages: List of message objects to clear.
        """
        for msg in messages:
            if isinstance(msg, list):
                msg.clear()
        messages.clear()

    @abstractmethod
    def generate_inner(self, inputs, **kwargs):
        """Generate response for given inputs. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement generate_inner method")

    def encode_image_directly(self, img_item):
        """Encode image directly without caching.

        Args:
            img_item: PIL Image object or other image data.

        Returns:
            Base64 encoded string or original data.
        """
        if isinstance(img_item, Image.Image):
            # Encode directly without caching
            try:
                return encode_image_to_base64(img_item)
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Image encoding failed: {str(e)}")
                raise
        elif isinstance(img_item, str) and img_item.startswith("data:image"):
            # Already in base64 format, return directly
            return img_item
        else:
            # Other cases (e.g., file path), open and encode
            img = Image.open(img_item)
            return self.encode_image_directly(img)

    def check_content(self, msgs):
        """Check input content type.

        Args:
            msgs: Raw input messages.

        Returns:
            str: Message type (str/dict/liststr/listdict).
        """
        if isinstance(msgs, str):
            return "str"
        if isinstance(msgs, dict):
            return "dict"
        if isinstance(msgs, list):
            types = [self.check_content(m) for m in msgs]
            if all(t == "str" for t in types):
                return "liststr"
            if all(t == "dict" for t in types):
                return "listdict"
        return "unknown"

    def preproc_content(self, inputs):
        """Convert raw input messages to unified dict list format.

        Args:
            inputs: Raw input.

        Returns:
            list: Processed input message list.
        """
        if self.check_content(inputs) == "str":
            return [dict(type="text", value=inputs)]
        elif self.check_content(inputs) == "dict":
            assert "type" in inputs and "value" in inputs
            return [inputs]
        elif self.check_content(inputs) == "liststr":
            res = []
            for s in inputs:
                mime, pth = parse_file(s, self.image_key_name)
                if mime is None or mime == "unknown":
                    res.append(dict(type="text", value=s))
                else:
                    res.append(dict(type=mime.split("/")[0], value=pth))
            return res
        elif self.check_content(inputs) == "listdict":
            for item in inputs:
                assert "type" in item and "value" in item
                # Skip file parsing for system prompts (plain text)
                if item["type"] == "system":
                    continue
                mime, s = parse_file(item["value"], self.image_key_name)
                if mime is None:
                    assert item["type"] == "text", item["value"]
                else:
                    assert item["type"] in ["image", "image_url"]
                    item["type"] = mime
                    item["value"] = s
            return inputs
        else:
            return None

    def prepare_inputs(self, inputs):
        """Prepare message format for API interface.

        Args:
            inputs: Input message list.

        Returns:
            tuple: (formatted_messages, text_content).
        """
        input_msgs = []

        # Extract system prompt if present
        system_items = [x for x in inputs if x["type"] == "system"]
        inputs = [x for x in inputs if x["type"] != "system"]
        for sys_item in system_items:
            if self.openai_format:
                input_msgs.append(
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": sys_item["value"]}],
                    }
                )
            else:
                input_msgs.append(
                    {
                        "role": "system",
                        "content": [{"type": "text", "value": sys_item["value"]}],
                    }
                )

        # Check if input contains images
        has_images = np.sum(
            [x["type"] == "image" or x["type"] == "image_url" for x in inputs]
        )
        if has_images:
            # Process messages containing images
            text = ""
            img_list = []

            # Collect all text and images
            for msg in inputs:
                if msg["type"] == "text":
                    text += (
                        msg["value"].replace("<image>\n", " ").replace("<image>", " ")
                    )
                elif msg["type"] == "image" or msg["type"] == "image_url":
                    # Handle list of images
                    if isinstance(msg["value"], list):
                        for img_item in msg["value"]:
                            if img_item is None:
                                if self.logger:
                                    self.logger.warning("Skipping None image in list")
                                continue
                            encoded_img = self.encode_image_directly(img_item)
                            img_list.append([msg["type"], encoded_img])

                    # Handle single image
                    elif msg["value"] is not None:
                        encoded_img = self.encode_image_directly(msg["value"])
                        img_list.append([msg["type"], encoded_img])

            # Build content with images and text
            if self.openai_format:
                content = [{"type": "text", "text": text}]
            else:
                content = [{"type": "text", "value": text}]
            for img_url in img_list:
                img_type, img_url = img_url
                if self.openai_format and img_type == "image_url":
                    content.append({"type": img_type, img_type: {"url": img_url}})
                else:
                    content.append({"type": img_type, "value": img_url})

            input_msgs.append({"role": "user", "content": content})
        else:
            # Process text-only messages
            assert all([x["type"] == "text" for x in inputs])
            text = "\n".join(
                [
                    x["value"].replace("<image>\n", " ").replace("<image>", " ")
                    for x in inputs
                ]
            )
            if self.openai_format:
                input_msgs.append(
                    {"role": "user", "content": [{"type": "text", "text": text}]}
                )
            else:
                input_msgs.append(
                    {"role": "user", "content": [{"type": "text", "value": text}]}
                )

        return input_msgs, text

    def _parse_openai_response(self, data: dict) -> str:
        """Parse OpenAI-format API response content.

        Args:
            data: Parsed JSON response from OpenAI API.

        Returns:
            Extracted content string, with optional reasoning_content formatting.
        """
        message_obj = data.get("choices", [{}])[0].get("message", {})
        content = message_obj.get("content", self.fail_msg)
        reasoning_content = message_obj.get("reasoning_content")

        if self.return_dict:
            return load_str_to_dict(content)
        if reasoning_content:
            return (
                f"<think>\n{reasoning_content.strip()}\n</think>\n\n{content.strip()}"
            )
        return content

    def _build_openai_headers(self, api_key: str) -> dict:
        """Build HTTP headers for OpenAI-compatible API request.

        Args:
            api_key: API key for authorization. If empty, no Authorization header.

        Returns:
            Headers dict with Content-Type and optional Authorization.
        """
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def _calculate_retry_delay(self, attempt: int, max_delay: float = 2.0) -> float:
        """Calculate exponential backoff delay with full jitter.

        Args:
            attempt: Current retry attempt (0-indexed).
            max_delay: Maximum delay cap in seconds.

        Returns:
            Delay in seconds (uniform random in ``[0, min(base * 2^attempt, max_delay)]``).
        """
        base_delay = 0.1
        exp_delay = min(base_delay * (2**attempt), min(self.wait, max_delay))
        return rd.random() * exp_delay

    def process_single_message(self, message, all_kwargs):
        """Process a single message with retry logic.

        Args:
            message: Single processed message.
            all_kwargs: Arguments passed to generate_inner.

        Returns:
            Response content or fail_msg on failure.
        """
        try:
            message = self.pre_process(message)
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Preprocessing failed: {str(e)}")
            return self.fail_msg

        # Initialize variables to avoid UnboundLocalError
        ret_code = None
        response_struct = None
        response = None

        for i in range(self.retry):
            try:
                ret_code, response_struct, response = self.generate_inner(
                    message, **all_kwargs
                )

                if ret_code == 0 and response_struct and response_struct != "":
                    return response_struct
                else:
                    raise Exception("Invalid response")
            except Exception as e:
                # Use exponential backoff with jitter (faster than fixed wait)
                if i < self.retry - 1:
                    delay = self._calculate_retry_delay(i)
                    time.sleep(delay)

                if i + 1 > self.retry * 2 // 3 and self.logger:
                    self.logger.warning(
                        f"Attempt {i+1}/{self.retry} failed, RetCode: {ret_code}, "
                        f"Response: {response_struct}, Error: {type(e).__name__}: {str(e)[:200]}"
                    )

        response_str = (
            response.text
            if hasattr(response, "text") and response
            else str(response) if response else "No response"
        )
        if self.logger:
            self.logger.warning(
                f"Failed after {self.retry} attempts, RetCode: {ret_code}, Response: {response_str}"
            )
        return self.fail_msg

    def pre_process(self, msg):
        """Validate and preprocess raw input messages into API-ready format.

        Args:
            msg: Raw input message (str, dict, or list).

        Returns:
            list: Preprocessed message list in OpenAI format.

        Raises:
            AssertionError: If input type is unsupported.
        """
        assert self.check_content(msg) in [
            "str",
            "dict",
            "liststr",
            "listdict",
        ], f"Unsupported input type: {msg}"

        msg = self.preproc_content(msg)
        assert msg is not None and self.check_content(msg) == "listdict"

        for item in msg:
            assert (
                item["type"] in self.allowed_types
            ), f'Unsupported input type: {item["type"]}'

        # Prepare input messages
        processed_msg, _ = self.prepare_inputs(msg)
        return processed_msg

    def _pre_encode_images(self, messages):
        """Pre-encode all PIL images to base64 strings in parallel.

        Encodes images before multi-threaded API calls to avoid PIL
        thread-safety issues.  Uses ``id``-based deduplication to encode
        each unique image object only once.

        Args:
            messages: List of message dicts containing potential PIL images.
                Modified in place (PIL objects replaced with base64 strings).
        """
        # Collect unique images
        images_to_encode = {}  # id -> image object

        for msg in messages:
            if not isinstance(msg, list):
                continue
            for item in msg:
                if not isinstance(item, dict):
                    continue
                if item.get("type") in ("image", "image_url") and "value" in item:
                    value = item["value"]
                    if isinstance(value, list):
                        for img in value:
                            if isinstance(img, Image.Image):
                                images_to_encode[id(img)] = img
                    elif isinstance(value, Image.Image):
                        images_to_encode[id(value)] = value

        if not images_to_encode:
            return

        if self.logger:
            self.logger.info(f"Pre-encoding {len(images_to_encode)} images...")

        # Encode images in parallel
        img_items = list(images_to_encode.items())
        encoded_cache = {}

        def encode_task(item):
            img_id, img = item
            return img_id, self.encode_image_directly(img)

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
                futures = {
                    executor.submit(encode_task, item): item for item in img_items
                }

                with tqdm(
                    total=len(img_items), desc="Encoding images", leave=False
                ) as pbar:
                    for future in concurrent.futures.as_completed(futures):
                        img_id, encoded = future.result()
                        encoded_cache[img_id] = encoded
                        pbar.update(1)
        except RuntimeError:
            # Fallback: if thread creation still fails, encode sequentially
            if self.logger:
                self.logger.warning(
                    "ThreadPoolExecutor failed (OS thread limit?), "
                    "falling back to sequential image encoding."
                )
            for item in tqdm(
                img_items, desc="Encoding images (sequential)", leave=False
            ):
                img_id, encoded = encode_task(item)
                encoded_cache[img_id] = encoded
        finally:
            gc.collect()

        if self.logger:
            self.logger.info(
                "Image byte encoding complete, replacing image references in payloads..."
            )

        # Replace PIL images with encoded strings
        # Note: Do NOT close PIL images here - they may be shared references
        # reused by later pipeline stages. They will be garbage collected when
        # no longer referenced.
        for msg in messages:
            if not isinstance(msg, list):
                continue
            for item in msg:
                if not isinstance(item, dict):
                    continue
                if item.get("type") in ("image", "image_url") and "value" in item:
                    value = item["value"]
                    if isinstance(value, list):
                        new_values = []
                        for img in value:
                            if isinstance(img, Image.Image):
                                encoded = encoded_cache.get(id(img), img)
                                new_values.append(encoded)
                            else:
                                new_values.append(img)
                        item["value"] = new_values
                    elif isinstance(value, Image.Image):
                        encoded = encoded_cache.get(id(value), value)
                        item["value"] = encoded

        # Clear references
        images_to_encode.clear()
        encoded_cache.clear()

        if self.logger:
            self.logger.info("Image encoding complete!")

    def generate(self, messages, **kwargs):
        """Main method to generate responses.

        Args:
            messages: Input messages (list of message dicts).
            **kwargs: Additional arguments.

        Returns:
            list: Generated responses for each message.
        """
        all_kwargs = cp.deepcopy(self.default_kwargs)
        all_kwargs.update(kwargs)

        # Handle empty messages list
        if not messages:
            return []

        # Pre-encode images (thread-safe)
        self._pre_encode_images(messages)

        # Use dedicated thread pool to avoid resource leaks
        max_workers = min(self.thread_num, len(messages))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks with retry on thread creation failure
            futures = {}
            for i, msg in enumerate(messages):
                while True:
                    try:
                        future = executor.submit(
                            self.process_single_message, msg, all_kwargs
                        )
                        futures[future] = i
                        break
                    except RuntimeError as e:
                        if "can't start new thread" in str(e):
                            # Wait for some tasks to complete before submitting more
                            time.sleep(0.1)
                        else:
                            raise

            # Use tqdm to display progress
            results = [None] * len(futures)

            with tqdm(total=len(futures), desc="Processing API Requests") as pbar:
                for future in concurrent.futures.as_completed(futures):
                    index = futures[future]
                    results[index] = future.result()
                    pbar.update(1)

        # Release base64 data before next batch
        self._clear_message_buffers(messages)
        self._collect_memory(trim=True)

        return results

    def shutdown(self):
        """Shutdown and release resources."""
        pass
