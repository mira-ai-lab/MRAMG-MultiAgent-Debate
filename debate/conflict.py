def detect_conflict(text_ans, visual_ans):
    conflicts = []
    for c in text_ans["claims"]:
        if not supported_by_visual(c, visual_ans):
            conflicts.append(c)
    return conflicts