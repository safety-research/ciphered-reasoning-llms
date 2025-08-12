def letter_to_word_with_dot(s):

    d_mapping = {
        "A": "Apple",
        "B": "Busy",
        "C": "Candle",
        "D": "Dragon",
        "E": "Echo",
        "F": "Forest",
        "G": "Galaxy",
        "H": "Harbor",
        "I": "Island",
        "J": "Journey",
        "K": "Kite",
        "L": "Lantern",
        "M": "Meadow",
        "N": "North",
        "O": "Ocean",
        "P": "Prism",
        "Q": "Quartz",
        "R": "River",
        "S": "Sunset",
        "T": "Thunder",
        "U": "Universe",
        "V": "Velvet",
        "W": "Willow",
        "X": "Xylophone",
        "Y": "Yellow",
        "Z": "Zephyr",
    }

    l_chars = list(s)

    for i, char in enumerate(l_chars):
        char = char.upper()

        if char in d_mapping:
            l_chars[i] = d_mapping[char]
        else:
            raise ValueError(f"Invalid character: {char}")

    return ".".join(l_chars)
