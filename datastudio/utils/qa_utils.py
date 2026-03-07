"""QA pair extraction and counting utilities."""

from typing import Any, Dict, List, Optional


def get_qa_count(conversations: List[Dict]) -> int:
    """
    Get number of QA pairs in conversation.

    Args:
        conversations: List of conversation turns.

    Returns:
        Number of complete QA pairs.
    """
    return len(conversations) // 2


def calculate_qa_statistics(
    datas: List[Dict],
    batch_size: Optional[int] = None,
    adjust_batch_size: bool = True,
    logger: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Calculate QA statistics and optionally adjust batch size.

    Args:
        datas: List of data items with "conversations" field.
        batch_size: Original batch size to adjust.
        adjust_batch_size: Whether to adjust batch size based on avg QA count.
        logger: Logger for info messages.

    Returns:
        Dictionary with:
        - avg_qa_count: Average QA pairs per item
        - max_qa_count: Maximum QA pairs
        - total_qa_pairs: Total QA pairs
        - real_batch_size: Adjusted batch size
    """
    if not datas:
        return {
            "avg_qa_count": 0,
            "max_qa_count": 0,
            "total_qa_pairs": 0,
            "real_batch_size": batch_size or 1,
        }

    qa_counts = []
    total_qa_pairs = 0

    for item in datas:
        conversations = item.get("conversations", [])
        if conversations:
            qa_count = get_qa_count(conversations)
            qa_counts.append(qa_count)
            total_qa_pairs += qa_count
        else:
            qa_counts.append(0)

    avg_qa_count = total_qa_pairs / len(datas) if datas else 0
    max_qa_count = max(qa_counts) if qa_counts else 0

    # Adjust batch size based on average QA count
    real_batch_size = batch_size
    if adjust_batch_size and batch_size is not None:
        expansion_factor = max(1.0, avg_qa_count)
        real_batch_size = max(1, int(batch_size / expansion_factor))

        if logger:
            logger.info(
                f"Batch size adjusted: original={batch_size}, "
                f"factor={expansion_factor:.2f}, adjusted={real_batch_size}"
            )

    return {
        "avg_qa_count": avg_qa_count,
        "max_qa_count": max_qa_count,
        "total_qa_pairs": total_qa_pairs,
        "real_batch_size": real_batch_size or 1,
    }
