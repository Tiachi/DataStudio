"""Schema converter between conversation data formats.

Converts between:
- **Standard**: ``{"conversations": [{"from": "human/gpt", "value": ...}], "image": ...}``
- **Messages**: ``{"messages": [{"role": "user/assistant", "content": ...}], "images": [...]}``
"""

from typing import Dict, List


class SchemaType:
    """Supported schema types."""

    STANDARD = "standard"  # conversations + image
    MESSAGES = "messages"  # messages + images


# Field mappings between schemas
ROLE_MAP_TO_STANDARD = {
    "user": "human",
    "assistant": "gpt",
    "system": "system",
}

ROLE_MAP_TO_MESSAGES = {
    "human": "user",
    "gpt": "assistant",
    "system": "system",
}


class SchemaConverter:
    """Automatic schema detection and conversion.

    Usage:
        # Load and normalize
        data = json.load(f)
        normalized = SchemaConverter.normalize(data)

        # Process data...

        # Save back to original format
        original = SchemaConverter.denormalize(normalized)
    """

    # Internal metadata key for storing original schema
    SCHEMA_KEY = "_original_schema"

    @classmethod
    def detect_schema(cls, item: Dict) -> str:
        """Detect the schema type of a data item.

        Args:
            item: A single data item dict.

        Returns:
            Schema type string (SchemaType.STANDARD or SchemaType.MESSAGES).
        """
        if "messages" in item:
            return SchemaType.MESSAGES
        elif "conversations" in item:
            return SchemaType.STANDARD
        else:
            # Default to standard if no conversation field found
            return SchemaType.STANDARD

    @classmethod
    def detect_schema_from_list(cls, data: List[Dict]) -> str:
        """Detect schema from a list of items (uses first valid item).

        Args:
            data: List of data items.

        Returns:
            Detected schema type.
        """
        for item in data:
            if "messages" in item or "conversations" in item:
                return cls.detect_schema(item)
        return SchemaType.STANDARD

    @classmethod
    def normalize_item(cls, item: Dict) -> Dict:
        """Convert a single item to standard schema.

        Stores original schema info for later denormalization.

        Args:
            item: Data item in any supported schema.

        Returns:
            Data item in standard schema with original schema metadata.
        """
        schema = cls.detect_schema(item)

        if schema == SchemaType.STANDARD:
            # Already standard, just mark it
            result = item.copy()
            result[cls.SCHEMA_KEY] = SchemaType.STANDARD
            return result

        elif schema == SchemaType.MESSAGES:
            return cls._messages_to_standard(item)

        return item

    @classmethod
    def denormalize_item(cls, item: Dict) -> Dict:
        """Convert a single item back to its original schema.

        Args:
            item: Data item in standard schema with schema metadata.

        Returns:
            Data item in original schema.
        """
        original_schema = item.get(cls.SCHEMA_KEY, SchemaType.STANDARD)

        # Create a copy without the schema key
        result = {k: v for k, v in item.items() if k != cls.SCHEMA_KEY}

        if original_schema == SchemaType.STANDARD:
            return result
        elif original_schema == SchemaType.MESSAGES:
            return cls._standard_to_messages(result)

        return result

    @classmethod
    def normalize(cls, data: List[Dict]) -> List[Dict]:
        """Normalize a list of items to standard schema.

        Args:
            data: List of data items.

        Returns:
            List of normalized items.
        """
        return [cls.normalize_item(item) for item in data]

    @classmethod
    def denormalize(cls, data: List[Dict]) -> List[Dict]:
        """Denormalize a list of items back to original schema.

        Args:
            data: List of normalized items.

        Returns:
            List of items in their original schemas.
        """
        return [cls.denormalize_item(item) for item in data]

    @classmethod
    def _messages_to_standard(cls, item: Dict) -> Dict:
        """Convert messages schema to standard schema.

        Input format:
            {
                "messages": [
                    {"role": "user", "content": "..."},
                    {"role": "assistant", "content": "..."}
                ],
                "images": ["path1.jpg", "path2.jpg"],
                ...
            }

        Output format:
            {
                "conversations": [
                    {"from": "human", "value": "..."},
                    {"from": "gpt", "value": "..."}
                ],
                "image": "path.jpg" or ["path1.jpg", ...],
                "_original_schema": "messages",
                ...
            }
        """
        result = {}

        # Convert messages -> conversations
        if "messages" in item:
            conversations = []
            for msg in item["messages"]:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                conversations.append(
                    {
                        "from": ROLE_MAP_TO_STANDARD.get(role, role),
                        "value": content,
                    }
                )
            result["conversations"] = conversations

        # Convert images -> image
        if "images" in item:
            images = item["images"]
            if isinstance(images, list):
                if len(images) == 1:
                    result["image"] = images[0]
                elif len(images) > 1:
                    result["image"] = images
                # Empty list: no image field
            else:
                result["image"] = images

        # Copy other fields (except messages and images)
        for key, value in item.items():
            if key not in ("messages", "images"):
                result[key] = value

        # Mark original schema
        result[cls.SCHEMA_KEY] = SchemaType.MESSAGES

        return result

    @classmethod
    def _standard_to_messages(cls, item: Dict) -> Dict:
        """Convert standard schema back to messages schema.

        Input format:
            {
                "conversations": [
                    {"from": "human", "value": "..."},
                    {"from": "gpt", "value": "..."}
                ],
                "image": "path.jpg" or ["path1.jpg", ...],
                ...
            }

        Output format:
            {
                "messages": [
                    {"role": "user", "content": "..."},
                    {"role": "assistant", "content": "..."}
                ],
                "images": ["path1.jpg", ...],
                ...
            }
        """
        result = {}

        # Convert conversations -> messages
        if "conversations" in item:
            messages = []
            for conv in item["conversations"]:
                from_role = conv.get("from", "human")
                value = conv.get("value", "")
                messages.append(
                    {
                        "role": ROLE_MAP_TO_MESSAGES.get(from_role, from_role),
                        "content": value,
                    }
                )
            result["messages"] = messages

        # Convert image -> images (always as list for messages schema)
        if "image" in item:
            image = item["image"]
            if isinstance(image, list):
                result["images"] = image
            elif image:
                result["images"] = [image]

        # Copy other fields (except conversations and image)
        for key, value in item.items():
            if key not in ("conversations", "image"):
                result[key] = value

        return result


def normalize_data(data: List[Dict]) -> List[Dict]:
    """Convenience function to normalize data.

    Args:
        data: List of data items in any supported schema.

    Returns:
        List of items in standard schema.
    """
    return SchemaConverter.normalize(data)


def denormalize_data(data: List[Dict]) -> List[Dict]:
    """Convenience function to denormalize data.

    Args:
        data: List of normalized items.

    Returns:
        List of items in their original schemas.
    """
    return SchemaConverter.denormalize(data)
