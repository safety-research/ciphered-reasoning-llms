import random


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
            pass

    return ".".join(l_chars)


def inverse_letter_to_word_with_dot(s):
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
    d_mapping = {v: k for k, v in d_mapping.items()}

    words = s.split(".")
    ret = ""
    for word in words:
        ret += d_mapping.get(word, word)

    return ret


def dot_between_chars(s):
    l_chars = list(s)

    return ".".join(l_chars)


def space_between_chars(s):
    l_chars = list(s)

    return " ".join(l_chars)


def inverse_space_between_chars(s):
    s = s.replace("   ", " ___ ")

    letters = s.split(" ")
    letters = [l.replace("___", " ") for l in letters]
    return "".join(letters)


# TODO(sguo35): do space between tokens


def inverse_dot_between_chars(s):
    letters = s.split(".")

    return "".join(letters)


def letter_to_poem_first_letter(s):
    poems_by_letter = {
        "A": [
            "Amber skies dissolve into a sleepy sea.",
            "Ancient winds hum secrets to the trees.",
            "A single drop of rain awakens the earth.",
            "Across the hills, the dawn begins to sing.",
            "An old oak whispers tales to the moon.",
            "Among the stars, silence feels alive.",
            "All rivers dream of meeting the ocean.",
            "A fragile petal braves the winter frost.",
        ],
        "B": [
            "Beneath the willow, shadows weave their dreams.",
            "Bright lanterns dance on the river’s skin.",
            "Brave hearts bloom in seasons of despair.",
            "Between the clouds, a shy sun peeks through.",
            "Bells of twilight ring in silver air.",
            "Bare feet find comfort in warm, soft earth.",
            "Birdsong pours like honey through the dawn.",
            "Breezes cradle the flowers into sleep.",
        ],
        "C": [
            "Crimson leaves drift on the evening tide.",
            "Calm waters hold the moon in their arms.",
            "Clouds wander slowly, thinking of rain.",
            "Candles flicker like thoughts half-formed.",
            "Crows carry messages the wind won’t tell.",
            "Clocks cannot measure a lover’s breath.",
            "Cold stars keep their distance from the fire.",
            "Crystal frost paints lace upon the glass.",
        ],
        "D": [
            "Dewdrops gather like secrets on the grass.",
            "Dark forests hum with unseen life.",
            "Distant hills wear crowns of quiet snow.",
            "Dreams arrive on the wings of night.",
            "Dancing flames speak in forgotten tongues.",
            "Delicate streams braid silver through the valley.",
            "Dusty roads remember every traveler’s step.",
            "Dusk folds the world in violet cloth.",
        ],
        "E": [
            "Echoes linger long after the song is gone.",
            "Emerald meadows hum under golden sun.",
            "Each star knows the name of the ocean.",
            "Evening leans gently on the tired day.",
            "Eyes of the earth close at midnight.",
            "Endless skies cradle the restless wind.",
            "Every raindrop writes its own short poem.",
            "Earth sighs beneath the weight of spring.",
        ],
        "F": [
            "Fallen leaves whisper of passing time.",
            "Fog curls like smoke over the river.",
            "Fireflies stitch light into the dark.",
            "Fields sway in the rhythm of the breeze.",
            "Footsteps vanish into the drifting snow.",
            "Fountains laugh in the heat of noon.",
            "Feathers float on the slow, wide stream.",
            "Frost bites gently at the morning grass.",
        ],
        "G": [
            "Golden light spills over the sleeping hills.",
            "Gentle rains sing to the budding earth.",
            "Gulls cry above the salted wind.",
            "Green vines climb toward a patient sky.",
            "Glass ripples catch the last of the sun.",
            "Gardens hold the breath of summer.",
            "Ghostly mist slips between the pines.",
            "Glowing coals breathe warmth into the dark.",
        ],
        "H": [
            "Hollow winds wander through ancient stone.",
            "Heavy clouds cradle the promise of rain.",
            "Hope blooms in the cracks of the pavement.",
            "Hushed waters kiss the edge of the shore.",
            "Horizon blushes at the touch of dawn.",
            "Hands of time scatter petals in the air.",
            "Hunters of stars prowl the velvet night.",
            "Hearts rest in the glow of the hearth.",
        ],
        "I": [
            "Ice sings softly under winter’s sun.",
            "In the shadows, silence takes shape.",
            "Ivory clouds drift in sleepy skies.",
            "Ink flows where the soul finds voice.",
            "In stillness, mountains hold their breath.",
            "Islands float like thoughts on the horizon.",
            "Illusions melt when the morning comes.",
            "In the rain, the earth begins again.",
        ],
        "J": [
            "Jade rivers carve through silent stone.",
            "Jasmine scents the hush of twilight.",
            "Joy hides in the rhythm of the rain.",
            "Journeys end where love begins again.",
            "Jewels of dew crown the morning grass.",
            "Jars of sunlight line the garden path.",
            "Justice walks slow, but never tires.",
            "Juniper winds sweep the sleeping hills.",
        ],
        "K": [
            "Kindness blooms in the coldest winters.",
            "Kites rise where the wind feels free.",
            "Keepsakes hide in the folds of time.",
            "Kings of the forest stand in quiet pride.",
            "Kisses fall like rain upon warm skin.",
            "Knots of ivy hold the old walls close.",
            "Knowledge flows where rivers meet the sea.",
            "Keen eyes trace the path of falling stars.",
        ],
        "L": [
            "Light spills over the rim of the earth.",
            "Laughter dances in the warm night air.",
            "Leaves rustle secrets to the ground.",
            "Lonely roads hum under fading light.",
            "Lamps glow like gentle moons in windows.",
            "Lingering rain sings on tin roofs.",
            "Long shadows reach toward the sea.",
            "Lilies dream in the still water.",
        ],
        "M": [
            "Moonlight drips from the edges of clouds.",
            "Mountains hum with the weight of ages.",
            "Morning spills gold into the valley.",
            "Mist curls low between sleeping trees.",
            "Memories rest in the scent of pine.",
            "Meadows sway in the breath of spring.",
            "Music drifts from unseen hands.",
            "Midnight wraps the world in velvet.",
        ],
        "N": [
            "Night leans softly on the world.",
            "Narrow streams wind through silver grass.",
            "Noon hums with the buzz of bees.",
            "New leaves shiver in the morning air.",
            "North winds speak of distant snow.",
            "Notes of a song linger in the dark.",
            "Nests cradle the promise of tomorrow.",
            "Nebulas bloom in the ocean of stars.",
        ],
        "O": [
            "Ocean tides hum the earth to sleep.",
            "Old walls sigh with the weight of years.",
            "Olive leaves whisper to the wind.",
            "Only the moon keeps secrets well.",
            "Opal skies glow at the edge of night.",
            "On the horizon, dreams begin to rise.",
            "Oaks stand tall through storm and sun.",
            "Over the hills, shadows slowly grow.",
        ],
        "P": [
            "Petals drift on the breath of spring.",
            "Paths wind deep into quiet woods.",
            "Pebbles sing beneath the stream’s flow.",
            "Peace settles on the cooling earth.",
            "Pages turn in the hands of time.",
            "Pines sway in the voice of the wind.",
            "Promises glow in the light of dawn.",
            "Painted skies close the day in fire.",
        ],
        "Q": [
            "Quiet rain seeps into thirsty soil.",
            "Quivering leaves wait for the wind.",
            "Questions linger like the scent of rain.",
            "Quilted clouds cover the morning sky.",
            "Queenly roses bow to the passing storm.",
            "Quartz shines in the sleeping stream.",
            "Quills scratch secrets into the night.",
            "Quick shadows dart across the field.",
        ],
        "R": [
            "Rivers hum to the heart of the forest.",
            "Raindrops stitch silver on the glass.",
            "Rustling wheat bends to the wind’s will.",
            "Roots drink deep from the earth’s veins.",
            "Red dawn spills fire across the sea.",
            "Restless tides chase the waning moon.",
            "Roses breathe perfume into the dusk.",
            "Rolling hills sleep under a clouded sky.",
        ],
        "S": [
            "Stars spill diamonds across the night.",
            "Soft snow hushes the restless earth.",
            "Shadows stretch in the last of the light.",
            "Songs drift through the open window.",
            "Silence blooms in the empty hall.",
            "Sunlight warms the edges of the day.",
            "Seashells hold whispers of the deep.",
            "Smoke curls lazy into the evening air.",
        ],
        "T": [
            "Twilight folds the day into dream.",
            "Tall grass bows in the weight of dew.",
            "Tides breathe slow against the shore.",
            "Trees cradle nests in their green arms.",
            "Tender hands shape the clay of time.",
            "Thunder rolls like ancient drums.",
            "Thin mist dances over the field.",
            "Tomorrow waits in the hush of night.",
        ],
        "U": [
            "Under the moon, the world feels still.",
            "Unseen rivers sing through the earth.",
            "Umbrellas bloom against the rain.",
            "Upward, the stars call out in silver.",
            "Untold stories hum in the wind.",
            "Upon the shore, footprints fade away.",
            "Under soft snow, seeds dream of spring.",
            "Unity hums in the song of the forest.",
        ],
        "V": [
            "Velvet skies hold the weight of stars.",
            "Vines curl in the sun’s slow dance.",
            "Voices echo through empty streets.",
            "Valleys rest beneath a silver mist.",
            "Violets nod to the morning light.",
            "Vision drifts far beyond the hills.",
            "Vanishing trails lead into the wild.",
            "Verdant hills sing in the summer wind.",
        ],
        "W": [
            "Waves speak softly to the sleeping sand.",
            "Wildflowers burn bright in the meadow.",
            "Whispers slip through the wooden door.",
            "Winter sighs over the frozen lake.",
            "Wind hums low in the shadowed pines.",
            "Warm rain kisses the dusty road.",
            "Winding rivers cradle the sunset.",
            "Wings beat steady against the dusk.",
        ],
        "X": [
            "Xylographs tell the story of old trees.",
            "Xenial smiles light the village square.",
            "Xyloid roots cling to the riverbank.",
            "Xanthic fields glow under the noon sun.",
            "Xylophones echo through the summer fair.",
            "Xeric winds sweep the barren dunes.",
            "Xenon lamps guard the lonely pier.",
            "Xyst paths gleam after the rain.",
        ],
        "Y": [
            "Yellow dawn spills warmth on the land.",
            "Young leaves sway in the spring air.",
            "Yearning drifts with the evening tide.",
            "Yesterdays fade into the mist.",
            "Yielding branches sway to the breeze.",
            "Yonder hills sleep in purple shade.",
            "Yarns of gold weave through the fields.",
            "Yew trees guard the ancient path.",
        ],
        "Z": [
            "Zephyrs dance in the scented grove.",
            "Zinnias glow beneath the morning sun.",
            "Zodiac skies wheel slow above the hills.",
            "Zones of light shift across the sea.",
            "Zero stars hide in the storm’s thick cloth.",
            "Zither notes drift through the summer air.",
            "Zinc roofs sing in the pelting rain.",
            "Zenith crowns the day in gold.",
        ],
    }

    l_chars = list(s)

    for i, char in enumerate(l_chars):
        char = char.upper()

        if char in poems_by_letter:
            random.seed(i)
            l_chars[i] = random.choice(poems_by_letter[char])
        else:
            pass

    return " ".join(l_chars)
