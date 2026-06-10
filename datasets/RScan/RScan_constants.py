### Change Labels 
VALID_CHANGE_IDS = [
    0, 
    1,
    2,
    3,
    4,
    5,
]

CHANGE_LABELS = [
    "static",
    "rigid", 
    "nonrigid", 
    "ambiguities", 
    "added", 
    "removed"]

RIO_CHANGE_COLOR_MAP = {
    0: (0.0, 0.0, 0.0),
    1: (174.0, 199.0, 232.0),
    2: (152.0, 223.0, 138.0),
    3: (31.0, 119.0, 180.0),
    4: (255.0, 187.0, 120.0),
    5: (188.0, 189.0, 34.0)
}



### Mapping NYU40 labels to ScanNet Benchmark constants ###
VALID_CLASS_IDS_20 = (
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    14,
    16,
    24,
    28,
    33,
    34,
    36,
    39,
)

CLASS_LABELS_20 = (
    "wall",
    "floor",
    "cabinet",
    "bed",
    "chair",
    "sofa",
    "table",
    "door",
    "window",
    "bookshelf",
    "picture",
    "counter",
    "desk",
    "curtain",
    "refrigerator",
    "shower curtain",
    "toilet",
    "sink",
    "bathtub",
    "otherfurniture",
)


"""
Mapping between 3RScan global IDs and ScanNet200 labels, including non-rigid category identification.
"""

# Direct mapping from 3RScan global IDs to ScanNet200 labels
GLOBAL_TO_SCANNET200 = {
    0: 0, # unlabelled --> unlabelled 
    1: 139, # air conditioner --> machine: generic machine category is closest match
    2: 32, # apron --> clothes: type of clothing garment
    3: 98, # aquarium --> container: container for holding water and fish
    4: 23, # armchair --> armchair: exact semantic match
    5: 99, # armoire --> wardrobe: direct semantic equivalent for large furniture piece for storage
    6: 250, # armor --> decoration: if displayed as decorative piece
    7: 118, # audio system --> speaker: closest match for sound equipment
    8: 11, # baby bed --> bed: type of bed
    9: 4, # baby changing table --> table: functional table surface
    10: 4, # baby changing unit --> table: functional table surface
    11: 1178, # baby gym --> structure: fixed exercise equipment
    12: 2, # baby seat --> chair: seating furniture piece
    13: 304, # baby toys --> stuffed animal: closest match for children's playthings
    14: 48, # backpack --> backpack: exact semantic match
    15: 47, # bag --> bag: exact semantic match
    16: 1178, # balcony --> structure: architectural element
    17: 5, # balcony door --> door: type of door
    18: 168, # ball --> ball: exact semantic match
    19: 145, # bar --> bar: exact semantic match
    20: 38, # bar stool --> stool: type of stool
    21: 177, # barrel --> water cooler: TODO maybe not?
    22: 38, # barstool --> stool: alternative spelling of bar stool
    23: 14, # basin --> sink: bathroom sink equivalent
    24: 73, # basket --> basket: exact semantic match
    25: 1173, # bath cabinet --> bathroom cabinet: direct equivalent
    26: 1156, # bath counter --> bathroom counter: direct equivalent
    27: 87, # bath rack --> rack: storage framework
    28: 32, # bath robe --> clothes: type of clothing
    29: 32, # bathrobe --> clothes: alternative spelling of bath robe
    30: 1163, # bathroom items --> object: generic category
    31: 42, # bathtub --> bathtub: exact semantic match
    32: 139, # bbq --> machine: cooking equipment
    33: 1178, # beam --> structure: structural building element
    34: 67, # bean bag --> ottoman: closest furniture match for casual seating
    35: 67, # beanbag --> ottoman: alternative spelling of bean bag
    36: 98, # beautician --> container: if storage unit
    37: 11, # bed --> bed: exact semantic match
    38: 34, # bed table --> nightstand: bedside table equivalent
    39: 34, # bedside table --> nightstand: exact semantic match
    40: 68, # bench --> bench: exact semantic match
    41: 233, # beverage crate --> crate: type of crate
    42: 121, # bicycle --> bicycle: exact semantic match
    43: 17, # bidet --> toilet: bathroom fixture most similar
    44: 121, # bike --> bicycle: alternative name for bicycle
    45: 66, # bin --> bin: exact semantic match
    46: 89, # blackboard --> blackboard: exact semantic match
    47: 54, # blanket --> blanket: exact semantic match
    48: 86, # blinds --> blinds: exact semantic match
    49: 69, # board --> board: exact semantic match
    50: 1163, # body loofah --> object: generic category
    51: 1163, # boiler --> object: if portable unit, pot
    52: 22, # book --> book: exact semantic match
    53: 22, # books --> book: plural of book
    54: 18, # bookshelf --> bookshelf: exact semantic match
    55: 63, # boots --> shoe: type of footwear
    56: 65, # bottle --> bottle: exact semantic match
    57: 65, # bottles --> bottle: plural of bottle
    58: 562, # bowl --> bowl: exact semantic match
    59: 26, # box --> box: exact semantic match
    60: 26, # boxes --> box: plural of box
    61: 1163, # bread --> object: food item
    62: 69, # breadboard --> board: type of kitchen board
    63: 79, # brochure --> paper: type of paper material
    64: 1163, # brush --> object: cleaning/grooming tool
    65: 102, # bucket --> bucket: exact semantic match
    66: 214, # buggy --> cart: wheeled transport device
    67: 154, # bulletin board --> bulletin board: exact semantic match
    68: 7, # cabinet --> cabinet: exact semantic match
    69: 1163, # cable --> object: electrical/utility item
    70: 87, # cable rack --> rack: storage framework
    71: 1187, # calendar --> calendar: exact semantic match
    72: 98, # can --> container: storage vessel
    73: 286, # candle --> candle: exact semantic match
    74: 286, # candles --> candle: plural of candle
    75: 250, # candlestick --> decoration: decorative holder
    76: 1178, # canopy --> structure: overhead covering
    77: 169, # cap --> hat: head covering
    78: 140, # carpet --> mat: floor covering
    79: 214, # carriage --> cart: wheeled transport device
    80: 214, # cart --> cart: exact semantic match
    81: 98, # case --> container: storage unit
    82: 41, # ceiling --> ceiling: exact semantic match
    83: 41, # ceiling /other room --> ceiling: alternative label for ceiling
    84: 1168, # ceiling light --> ceiling light: exact semantic match
    85: 2, # chair --> chair: exact semantic match
    86: 2, # chairs --> chair: plural of chair
    87: 105, # chandelier --> light: lighting fixture
    88: 4, # changing table --> table: functional surface
    89: 221, # chest --> storage container: large storage unit
    90: 2, # child chair --> chair: children's seating
    91: 32, # child clothes --> clothes: children's garments
    92: 4, # children's table --> table: children's furniture
    93: 1163, # cleaning agent --> object: cleaning supply
    94: 1163, # cleaning brush --> object: cleaning tool
    95: 1163, # cleanser --> object: cleaning supply
    96: 103, # clock --> clock: exact semantic match
    97: 57, # closet --> closet: exact semantic match
    98: 276, # closet door --> closet door: exact semantic match
    99: 31, # cloth --> towel: closest match for fabric material
    100: 32, # clothes --> clothes: direct semantic match
    101: 110, # clothes dryer --> clothes dryer: direct semantic match
    102: 87, # clothes rack --> rack: semantic match as it's a structure for holding clothes
    103: 1163, # clutter --> object: generic catch-all category for mixed items
    104: 131, # coat --> jacket: semantic match for outerwear
    105: 1163, # coffee --> object: as a raw material
    106: 134, # coffee machine --> coffee maker: direct semantic equivalent
    107: 134, # coffee maker --> coffee maker: direct semantic match
    108: 24, # coffee table --> coffee table: direct semantic match
    109: 120, # column --> column: direct semantic match
    110: 36, # commode --> dresser: most similar as it's a tall chest of drawers
    111: 64, # computer --> computer tower: semantic match for desktop computer
    112: 9, # computer desk --> desk: semantic match for workspace furniture
    113: 51, # console --> tv stand: if used for entertainment
    114: 98, # container --> container: direct semantic match
    115: 1163, # cooking pot --> object: as a kitchen implement
    116: 68, # corner bench --> bench: semantic match for seating
    117: 98, # cosmetics kit --> container: as it holds items
    118: 6, # couch --> couch: direct semantic match
    119: 44, # couch table --> end table: semantic match for side table
    120: 35, # counter --> counter: direct semantic match
    121: 54, # cover --> blanket: semantic match for bed covering
    122: 11, # cradle --> bed: as it's for sleeping
    123: 233, # crate --> crate: direct semantic match
    124: 11, # crib --> bed: semantic match as sleeping furniture
    125: 1163, # cube --> object: if decorative/other purpose
    126: 130, # cup --> cup: direct semantic match
    127: 7, # cupboard --> cabinet: semantic match for storage furniture
    128: 130, # cups --> cup: direct semantic match (plural form)
    129: 21, # curtain --> curtain: direct semantic match
    130: 170, # curtain rail --> shower curtain rod: semantic match for curtain support structure
    131: 39, # cushion --> cushion: direct semantic match
    132: 39, # cushions stack --> cushion: semantic match (multiple cushions)
    133: 69, # cut board --> board: semantic match for cutting surface
    134: 69, # cutting board --> board: semantic match for cutting surface
    135: 139, # cycling trainer --> machine: as exercise equipment
    136: 250, # darts --> decoration: if mounted as wall art
    137: 250, # decoration --> decoration: direct semantic match
    138: 9, # desk --> desk: direct semantic match
    139: 10, # desk chair --> office chair: semantic match for desk seating
    140: 139, # device --> machine: semantic match for electronic/mechanical items
    141: 1163, # diapers --> object: as personal care item
    142: 2, # dining chair --> chair: semantic match for dining seating
    143: 45, # dining set --> dining table: semantic match for dining furniture set
    144: 45, # dining table --> dining table: direct semantic match
    145: 1174, # discs --> cd case: if computer case
    146: 88, # dish --> plate: semantic match for dining vessel
    147: 323, # dish dryer --> dish rack: semantic match for drying dishes
    148: 323, # dishdrainer --> dish rack: semantic match for drying dishes
    149: 88, # dishes --> plate: semantic match (plural form)
    150: 136, # dishwasher --> dishwasher: direct semantic match
    151: 100, # dispenser --> soap dispenser: if for soap
    152: 79, # documents --> paper: semantic match for written materials
    153: 1163, # dog --> object: if real/statue
    154: 304, # doll --> stuffed animal: semantic match for toys
    155: 5, # door --> door: direct semantic match
    156: 5, # door /other room --> door: direct semantic match
    157: 140, # door mat --> mat: semantic match for floor covering
    158: 161, # doorframe --> doorframe: direct semantic match
    159: 161, # doorframe /other room --> doorframe: direct semantic match
    160: 107, # drain pipe --> pipe: semantic match for plumbing
    161: 7, # drawer --> cabinet: semantic match as storage furniture component
    162: 7, # drawers --> cabinet: semantic match as storage furniture components
    163: 87, # drawers rack --> rack: semantic match for storage structure
    164: 32, # dress --> clothes: as clothing item
    165: 36, # dresser --> dresser: direct semantic match
    166: 1164, # dressing table --> bathroom vanity: if in bathroom
    167: 65, # drinks --> bottle: if in bottles
    168: 1163, # drum --> object: as musical instrument
    169: 110, # drying machine --> clothes dryer: semantic match for laundry appliance
    170: 87, # drying rack --> rack: semantic match for drying structure
    171: 1170, # dumbbells --> dumbbell: direct semantic match
    172: 1178, # elevator --> structure: as building component
    173: 139, # elliptical trainer --> machine: as exercise equipment
    174: 342, # exhaust hood --> range hood: direct semantic match
    175: 261, # exit sign --> sign: semantic match for informational display
    176: 76, # extractor fan --> fan: semantic match for air movement device
    177: 1163, # fabric --> object: if raw material
    178: 76, # fan --> fan: direct semantic match
    179: 748, # fence --> divider: as room separator
    180: 21, # festoon --> curtain: semantic match for decorative hanging
    181: 250, # figure --> decoration: if ornamental
    182: 75, # file cabinet --> file cabinet: direct semantic match
    183: 166, # fire extinguisher --> fire extinguisher: direct semantic match
    184: 156, # fireplace --> fireplace: direct semantic match
    185: 132, # firewood box --> storage bin: for holding wood
    186: 250, # flag --> decoration: if decorative
    187: 52, # flipchart --> whiteboard: as writing surface
    188: 3, # floor --> floor: direct semantic match
    189: 3, # floor /other room --> floor: direct semantic match
    190: 28, # floor lamp --> lamp: semantic match for lighting
    191: 140, # floor mat --> mat: semantic match for floor covering
    192: 40, # flower --> plant: semantic match for flora
    193: 40, # flowers --> plant: semantic match for flora (plural)
    194: 1163, # flush --> object: if control mechanism
    195: 1184, # folded beach chairs --> folded chair: semantic match for portable seating
    196: 98, # folder --> container: as storage item
    197: 1184, # folding chair --> folded chair: direct semantic match
    198: 1163, # food --> object: as consumable items
    199: 4, # foosball table --> table: semantic match as game furniture
    200: 67, # footstool --> ottoman: low seat for resting feet
    201: 15, # frame --> picture: support structure for pictures/mirrors
    202: 27, # fridge --> refrigerator: exact semantic match for cooling appliance
    203: 1163, # fruit --> object: food item
    204: 88, # fruit plate --> plate: serving dish
    205: 1163, # fruits --> object: plural of fruit
    206: 213, # furniture --> furniture: exact semantic match
    207: 1163, # garbage --> object: discarded material
    208: 56, # garbage bin --> trash can: waste container
    209: 213, # garden umbrella --> furniture: outdoor furniture
    210: 139, # generator --> machine: power generating device
    211: 130, # glass --> cup: drinking vessel
    212: 1178, # glass wall --> structure: transparent wall element
    213: 140, # grass --> mat: if artificial turf
    214: 112, # guitar --> guitar: exact semantic match
    215: 168, # gymnastic ball --> ball: exercise ball
    216: 370, # hair dryer --> hair dryer: exact semantic match
    217: 1163, # hand brush --> object: cleaning tool
    218: 370, # hand dryer --> hair dryer: similar function to hair dryer
    219: 100, # hand washer --> soap dispenser: cleaning product dispenser
    220: 399, # handbag --> purse: carrying accessory
    221: 395, # handhold --> handicap bar: support bar
    222: 1163, # handle --> object: apparatus for gripping
    223: 1171, # handrail --> stair rail: support rail
    224: 1163, # hanger --> object: clothes hanging device
    225: 1163, # hangers --> object: plural of hanger
    226: 7, # hanging cabinet --> cabinet: wall-mounted storage unit
    227: 213, # headboard --> furniture: bed component
    228: 96, # heater --> radiator: heating device
    229: 1163, # helmet --> object: protective gear
    230: 342, # hood --> range hood: kitchen ventilation
    231: 139, # humidifier --> machine: air treatment device
    232: 1163, # hygiene products --> object: bathroom supplies
    233: 1163, # instrument --> object: specialized tool/device
    234: 1163, # iron --> object: clothes pressing device
    235: 155, # ironing board --> ironing board: exact semantic match
    236: 1163, # item --> object: generic item
    237: 1163, # items --> object: plural of item
    238: 131, # jacket --> jacket: exact semantic match
    239: 86, # jalousie --> blinds: window covering
    240: 98, # jar --> container: storage vessel
    241: 98, # jug --> container: liquid container
    242: 139, # juicer --> machine: kitchen appliance
    243: 1176, # kettle --> coffee kettle: closest match for water heating vessel
    244: 46, # keyboard --> keyboard: exact semantic match
    245: 121, # kids bicycle --> bicycle: children's version of bicycle
    246: 2, # kids chair --> chair: children's version of chair
    247: 2, # kids rocking chair --> chair: specialized children's chair
    248: 38, # kids stool --> stool: children's version of stool
    249: 4, # kids table --> table: children's version of table
    250: 139, # kitchen appliance --> machine: general kitchen equipment
    251: 29, # kitchen cabinet --> kitchen cabinet: exact semantic match
    252: 159, # kitchen counter --> kitchen counter: exact semantic match
    253: 342, # kitchen hood --> range hood: ventilation system
    254: 1163, # kitchen item --> object: kitchen-specific item
    255: 1163, # kitchen object --> object: kitchen-specific item
    256: 1163, # kitchen playset --> object: children's toy
    257: 87, # kitchen rack --> rack: storage framework
    258: 14, # kitchen sink --> sink: exact semantic match
    259: 6, # kitchen sofa --> couch: kitchen seating
    260: 31, # kitchen towel --> towel: kitchen-specific towel
    261: 98, # knife box --> container: storage for utensils
    262: 122, # ladder --> ladder: exact semantic match
    263: 28, # lamp --> lamp: exact semantic match
    264: 77, # laptop --> laptop: exact semantic match
    265: 106, # laundry basket --> laundry basket: exact semantic match
    266: 79, # letter --> paper: written communication
    267: 105, # light --> light: exact semantic match
    268: 31, # linen --> towel: fabric item
    269: 221, # locker --> storage container: storage unit
    270: 221, # lockers --> storage container: plural of locker
    271: 11, # loft bed --> bed: elevated sleeping furniture
    272: 74, # lounger --> sofa chair: reclining chair
    273: 1190, # luggage --> luggage: exact semantic match
    274: 139, # machine --> machine: exact semantic match
    275: 79, # magazine --> paper: reading material
    276: 79, # magazine files --> paper: stored reading material
    277: 87, # magazine rack --> rack: display framework
    278: 104, # magazine stand --> stand: display support
    279: 1163, # mandarins --> object: food item
    280: 250, # mannequin --> decoration: display figure
    281: 250, # mask --> decoration: face covering
    282: 1191, # mattress --> mattress: exact semantic match
    283: 1163, # medical device --> object: if portable
    284: 79, # menu --> paper: restaurant listing
    285: 1163, # meter --> object: measuring device
    286: 59, # microwave --> microwave: exact semantic match
    287: 1163, # milk --> object: food item
    288: 71, # mirror --> mirror: exact semantic match
    289: 19, # monitor --> monitor: exact semantic match
    290: 1163, # mop --> object: cleaning tool
    291: 139, # multicooker --> machine: cooking device
    292: 138, # napkins --> paper towel roll: closest match for table napkins
    293: 79, # newspaper --> paper: printed material
    294: 87, # newspaper rack --> rack: display framework
    295: 34, # nightstand --> nightstand: exact semantic match
    296: 22, # notebook --> book: bound paper
    297: 77, # notebooks --> laptop: portable computer
    298: 1163, # object --> object: exact semantic match
    299: 1163, # objects --> object: plural of object
    300: 10, # office chair --> office chair: exact semantic match
    301: 9, # office table --> desk: table for office work
    302: 1163, # organizer --> object: personal-electronic
    303: 67, # ottoman --> ottoman: exact semantic match
    304: 84, # oven --> oven: exact semantic match
    305: 1163, # oven glove --> object: kitchen safety item
    306: 221, # pack --> storage container: container for items
    307: 221, # package --> storage container: wrapped container
    308: 221, # packs --> storage container: plural of pack
    309: 15, # painting --> picture: wall art
    310: 1163, # pan --> object: cooking vessel
    311: 79, # paper --> paper: exact semantic match
    312: 180, # paper cutter --> paper cutter: exact semantic match
    313: 115, # paper holder --> toilet paper holder: closest match for paper holding device
    314: 261, # paper sign --> sign: exact semantic match
    315: 79, # paper stack --> paper: collection of paper
    316: 138, # paper towel --> paper towel roll: exact semantic match
    317: 82, # paper towel dispenser --> paper towel dispenser: exact semantic match
    318: 79, # papers --> paper: plural of paper
    319: 748, # partition --> divider: room separator
    320: 3, # pavement --> floor: ground surface
    321: 64, # pc --> computer tower: desktop computer
    322: 1163, # pepper --> object: food seasoning
    323: 39, # pet bed --> cushion: animal sleeping surface
    324: 15, # photo frame --> picture: display frame
    325: 15, # photos --> picture: photographic images
    326: 90, # piano --> piano: exact semantic match
    327: 15, # picture --> picture: exact semantic match
    328: 15, # pictures --> picture: plural of picture
    329: 1163, # pile --> object: collection of stacked items
    330: 22, # pile of books --> book: stacked reading materials
    331: 65, # pile of bottles --> bottle: stacked containers
    332: 286, # pile of candles --> candle: stacked light sources
    333: 79, # pile of folders --> paper: stacked document holders
    334: 79, # pile of papers --> paper: stacked documents
    335: 13, # pile of pillows --> pillow: stacked cushions
    336: 1163, # pile of wires --> object: electrical components
    337: 191, # pillar --> pillar: exact semantic match
    338: 13, # pillow --> pillow: exact semantic match
    339: 154, # pin board wall --> bulletin board: wall mounting board
    340: 107, # pipe --> pipe: exact semantic match
    341: 69, # plank --> board: wooden board
    342: 40, # plant --> plant: exact semantic match
    343: 40, # planter --> plant: container for plants
    344: 40, # plants --> plant: plural of plant
    345: 88, # plate --> plate: exact semantic match
    346: 88, # plates --> plate: plural of plate
    347: 1178, # platform --> structure: raised surface
    348: 118, # player --> speaker: audio device
    349: 221, # pocket --> storage container: small storage space
    350: 1178, # podest --> structure: raised platform
    351: 304, # pooh --> stuffed animal: plush toy
    352: 1188, # poster --> poster: exact semantic match
    353: 98, # pot --> container: cooking vessel
    354: 1163, # price tag --> object: item label
    355: 50, # printer --> printer: exact semantic match
    356: 264, # projector --> projector: exact semantic match
    357: 39, # puf --> cushion: soft seating
    358: 304, # puppet --> stuffed animal: closest match for toy figure
    359: 87, # rack --> rack: exact semantic match
    360: 96, # radiator --> radiator: exact semantic match
    361: 118, # radio --> speaker: audio device
    362: 31, # rag --> towel: cleaning cloth
    363: 95, # rail --> rail: exact semantic match
    364: 95, # railing --> rail: safety barrier
    365: 1178, # ramp --> structure: inclined surface
    366: 97, # recycle bin --> recycling bin: exact semantic match
    367: 27, # refrigerator --> refrigerator: exact semantic match
    368: 2, # rocking chair --> chair: specialized chair type
    369: 138, # roll --> paper towel roll: rolled paper product
    370: 140, # rolled carpet --> mat: floor covering
    371: 214, # rolling cart --> cart: wheeled transport
    372: 1163, # rolling pin --> object: kitchen tool
    373: 1178, # roof --> structure: building top
    374: 4, # round table --> table: circular table
    375: 139, # rowing machine --> machine: exercise equipment
    376: 1169, # rubbish bin --> trash bin: waste container
    377: 140, # rug --> mat: floor covering
    378: 47, # sack --> bag: carrying container
    379: 1163, # salad --> object: food item
    380: 1163, # salt --> object: food seasoning
    381: 98, # sauce boat --> container: serving vessel
    382: 229, # scale --> scale: exact semantic match
    383: 32, # scarf --> clothes: clothing accessory
    384: 19, # screen --> monitor: display device
    385: 116, # seat --> seat: exact semantic match
    386: 39, # seat pad --> cushion: comfort padding
    387: 139, # sewing machine --> machine: crafting device
    388: 86, # shades --> blinds: window covering
    389: 1163, # shampoo --> object: cleaning product
    390: 54, # sheets --> blanket: bed covering
    391: 8, # shelf --> shelf: exact semantic match
    392: 1163, # shelf clutter --> object: miscellaneous items
    393: 169, # shelf of caps --> hat: stored headwear
    394: 8, # shelf unit --> shelf: storage furniture
    395: 8, # shelves --> shelf: plural of shelf
    396: 32, # shirt --> clothes: clothing item
    397: 63, # shoe --> shoe: exact semantic match
    398: 26, # shoe box --> box: footwear container
    399: 87, # shoe commode --> rack: footwear storage
    400: 87, # shoe rack --> rack: footwear storage frame
    401: 8, # shoe shelf --> shelf: footwear storage surface
    402: 63, # shoes --> shoe: plural of shoe
    403: 7, # showcase --> cabinet: display furniture
    404: 78, # shower --> shower: exact semantic match
    405: 55, # shower curtain --> shower curtain: exact semantic match
    406: 188, # shower door --> shower door: exact semantic match
    407: 417, # shower floor --> shower floor: exact semantic match
    408: 100,   # shower gel --> soap dispenser 
    409: 128, # shower wall --> shower wall: exact semantic match
    410: 44, # side table --> end table: exact semantic match
    411: 7, # sideboard --> cabinet: dining room storage
    412: 6, # sidecouch --> couch: seating furniture
    413: 44, # sidetable --> end table: alternative spelling of side table
    414: 261, # sign --> sign: exact semantic match
    415: 14, # sink --> sink: exact semantic match
    416: 35, # sink counter --> counter: sink surround
    417: 1, # slanted wall --> wall: angled wall structure
    418: 1163, # snowboard --> object: sports equipment
    419: 100, # soap --> soap dispenser: cleaning product
    420: 157, # soap dish --> soap dish: exact semantic match
    421: 100, # soap dispenser --> soap dispenser: exact semantic match
    422: 242, # socket --> power outlet: electrical connection
    423: 6, # sofa --> couch: exact semantic match
    424: 74, # sofa chair --> sofa chair: exact semantic match
    425: 6, # sofa couch --> couch: alternative name for sofa
    426: 118, # speaker --> speaker: exact semantic match
    427: 1163, # spice --> object: food seasoning
    428: 1163, # spices --> object: plural of spice
    429: 1163, # sponge --> object: cleaning tool
    430: 105, # spots --> light: lighting fixtures
    431: 1163, # squeezer --> object: kitchen tool
    432: 58, # stair --> stairs: single step
    433: 58, # stairs --> stairs: exact semantic match
    434: 104, # stand --> stand: exact semantic match
    435: 250, # star --> decoration: decorative element
    436: 250, # statue --> decoration: decorative sculpture
    437: 250, # statuette --> decoration: small decorative sculpture
    438: 122, # stepladder --> ladder: folding ladder type
    439: 118, # stereo --> speaker: audio equipment
    440: 118, # stereo equipment --> speaker: audio system
    441: 1163, # stick --> object: long thin item
    442: 38, # stool --> stool: exact semantic match
    443: 221, # storage --> storage container: general storage
    444: 132, # storage bin --> storage bin: exact semantic match
    445: 221, # storage box --> storage container: box for storage
    446: 221, # storage container --> storage container: exact semantic match
    447: 221, # storage unit --> storage container: larger storage solution
    448: 62, # stove --> stove: exact semantic match
    449: 214, # stroller --> cart: baby transport
    450: 304, # stuffed animal --> stuffed animal: exact semantic match
    451: 1163, # sugar packs --> object: food sweetener
    452: 93, # suitcase --> suitcase: exact semantic match
    453: 232, # switch --> light switch: exact semantic match
    454: 32, # t-shirt --> clothes: clothing item
    455: 4, # table --> table: exact semantic match
    456: 28, # table lamp --> lamp: exact semantic match
    457: 4, # table soccer --> table: game table
    458: 77, # tablet --> laptop: portable computing device
    459: 98, # teapot --> container: beverage container
    460: 304, # teddy bear --> stuffed animal: plush toy
    461: 101, # telephone --> telephone: exact semantic match
    462: 1163, # tennis raquet --> object: sports equipment
    463: 1163, # tent --> object: portable shelter
    464: 1163, # things --> object: generic items
    465: 1163, # tile --> object: surface covering
    466: 1163, # tire --> object: wheel component
    467: 230, # tissue pack --> tissue box: exact semantic match
    468: 148, # toaster --> toaster: exact semantic match
    469: 17, # toilet --> toilet: exact semantic match
    470: 1163, # toilet brush --> object: cleaning tool
    471: 49, # toilet paper --> toilet paper: exact semantic match
    472: 163, # toilet paper dispenser --> toilet paper dispenser: exact semantic match
    473: 115, # toilet paper holder --> toilet paper holder: exact semantic match
    474: 1163, # toiletry --> object: bathroom items
    475: 69, # tool wall --> board: mounting surface
    476: 31, # towel --> towel: exact semantic match
    477: 73, # towel basket --> basket: towel storage
    478: 31, # towels --> towel: plural of towel
    479: 304, # toy --> stuffed animal: plaything
    480: 1163, # toy house --> object: children's playset
    481: 1169, # trash bin --> trash bin: exact semantic match
    482: 56, # trash can --> trash can: exact semantic match
    483: 56, # trashcan --> trash can: alternative spelling
    484: 185, # tray --> tray: exact semantic match
    485: 139, # treadmill --> machine: exercise equipment
    486: 40, # tree --> plant: living plant
    487: 250, # tree decoration --> decoration: ornamental item
    488: 1172, # tube --> tube: exact semantic match
    489: 33, # tv --> tv: exact semantic match
    490: 51, # tv stand --> tv stand: exact semantic match
    491: 51, # tv table --> tv stand: table for television
    492: 139, # typewriter --> machine: typing device
    493: 112, # ukulele --> guitar: string instrument
    494: 1163, # umbrella --> object: rain protection
    495: 1, # upholstered wall --> wall: padded wall surface
    496: 17, # urinal --> toilet: bathroom fixture
    497: 1163, # utensils --> object: kitchen tools
    498: 283, # vacuum --> vacuum cleaner: cleaning device
    499: 283, # vacuum cleaner --> vacuum cleaner: exact semantic match
    500: 250, # vase --> decoration: ornamental container
    501: 408, # ventilation --> vent: air system
    502: 76, # ventilator --> fan: air circulation device
    503: 1, # wall --> wall: exact semantic match
    504: 1, # wall /other room --> wall: alternative label for wall
    505: 161, # wall frame --> doorframe: wall opening frame
    506: 40, # wall plants --> plant: mounted vegetation
    507: 87, # wall rack --> rack: wall-mounted storage
    508: 99, # wardrobe --> wardrobe: exact semantic match
    509: 276, # wardrobe door --> closet door: storage unit door
    510: 14, # washbasin --> sink: bathroom fixture
    511: 106, # washing basket --> laundry basket: clothes container
    512: 70, # washing machine --> washing machine: exact semantic match
    513: 1163, # washing powder --> object: cleaning product
    514: 392, # water --> waterbottle: liquid
    515: 1178, # water heater --> structure: heating system
    516: 488, # watering can --> water pitcher: plant care tool
    517: 1170, # weights --> dumbbell: exercise equipment
    518: 1170, # weighths --> dumbbell: alternative spelling of weights
    519: 52, # whiteboard --> whiteboard: exact semantic match
    520: 16, # window --> window: exact semantic match
    521: 141, # window board --> windowsill: window base
    522: 1163, # window clutter --> object: window items
    523: 161, # window frame --> doorframe: window structure
    524: 16, # windows --> window: plural of window
    525: 141, # windowsill --> windowsill: exact semantic match
    526: 1163, # wood --> object: building material
    527: 26, # wood box --> box: wooden container
    528: 64, # xbox --> computer tower: gaming device
}

def get_scannet200_label(global_id: int) -> str:
    """Get the corresponding ScanNet200 Id for a given global ID."""
    label = GLOBAL_TO_SCANNET200.get(global_id)
    if label is None:
        raise KeyError(f"Global ID {global_id} not found in mapping")
    return label

### For instance segmentation the non-object categories ###
VALID_PANOPTIC_IDS = (1, 3)

CLASS_LABELS_PANOPTIC = ("wall", "floor")