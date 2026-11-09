ACTIVE_KEY_IDS = ["whsec_2026_05"]

def accepts(key_id: str) -> bool:
    return key_id in ACTIVE_KEY_IDS
