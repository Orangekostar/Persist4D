# Define ScanNet200 constants
VALID_CLASS_IDS_200 = (
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 26, 27, 28, 29, 31, 32, 33, 34, 35, 36, 38, 39, 40, 41, 42, 44, 45, 46, 47, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 59, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71,
    72, 73, 74, 75, 76, 77, 78, 79, 80, 82, 84, 86, 87, 88, 89, 90, 93, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 110, 112, 115, 116, 118, 120, 121, 122, 125, 128, 130, 131, 132, 134, 136, 138, 139, 140, 141, 145, 148, 154,
    155, 156, 157, 159, 161, 163, 165, 166, 168, 169, 170, 177, 180, 185, 188, 191, 193, 195, 202, 208, 213, 214, 221, 229, 230, 232, 233, 242, 250, 261, 264, 276, 283, 286, 300, 304, 312, 323, 325, 331, 342, 356, 370, 392, 395, 399, 408, 417,
    488, 540, 562, 570, 572, 581, 609, 748, 776, 1156, 1163, 1164, 1165, 1166, 1167, 1168, 1169, 1170, 1171, 1172, 1173, 1174, 1175, 1176, 1178, 1179, 1180, 1181, 1182, 1183, 1184, 1185, 1186, 1187, 1188, 1189, 1190, 1191
)

CLASS_LABELS_200 = (
    'wall', 'chair', 'floor', 'table', 'door', 'couch', 'cabinet', 'shelf', 'desk', 'office chair', 'bed', 'pillow', 'sink', 'picture', 'window', 'toilet', 'bookshelf', 'monitor', 'curtain', 'book', 'armchair', 'coffee table', 'box',
    'refrigerator', 'lamp', 'kitchen cabinet', 'towel', 'clothes', 'tv', 'nightstand', 'counter', 'dresser', 'stool', 'cushion', 'plant', 'ceiling', 'bathtub', 'end table', 'dining table', 'keyboard', 'bag', 'backpack', 'toilet paper',
    'printer', 'tv stand', 'whiteboard', 'blanket', 'shower curtain', 'trash can', 'closet', 'stairs', 'microwave', 'stove', 'shoe', 'computer tower', 'bottle', 'bin', 'ottoman', 'bench', 'board', 'washing machine', 'mirror', 'copier',
    'basket', 'sofa chair', 'file cabinet', 'fan', 'laptop', 'shower', 'paper', 'person', 'paper towel dispenser', 'oven', 'blinds', 'rack', 'plate', 'blackboard', 'piano', 'suitcase', 'rail', 'radiator', 'recycling bin', 'container',
    'wardrobe', 'soap dispenser', 'telephone', 'bucket', 'clock', 'stand', 'light', 'laundry basket', 'pipe', 'clothes dryer', 'guitar', 'toilet paper holder', 'seat', 'speaker', 'column', 'bicycle', 'ladder', 'bathroom stall', 'shower wall',
    'cup', 'jacket', 'storage bin', 'coffee maker', 'dishwasher', 'paper towel roll', 'machine', 'mat', 'windowsill', 'bar', 'toaster', 'bulletin board', 'ironing board', 'fireplace', 'soap dish', 'kitchen counter', 'doorframe',
    'toilet paper dispenser', 'mini fridge', 'fire extinguisher', 'ball', 'hat', 'shower curtain rod', 'water cooler', 'paper cutter', 'tray', 'shower door', 'pillar', 'ledge', 'toaster oven', 'mouse', 'toilet seat cover dispenser',
    'furniture', 'cart', 'storage container', 'scale', 'tissue box', 'light switch', 'crate', 'power outlet', 'decoration', 'sign', 'projector', 'closet door', 'vacuum cleaner', 'candle', 'plunger', 'stuffed animal', 'headphones', 'dish rack',
    'broom', 'guitar case', 'range hood', 'dustpan', 'hair dryer', 'water bottle', 'handicap bar', 'purse', 'vent', 'shower floor', 'water pitcher', 'mailbox', 'bowl', 'paper bag', 'alarm clock', 'music stand', 'projector screen', 'divider',
    'laundry detergent', 'bathroom counter', 'object', 'bathroom vanity', 'closet wall', 'laundry hamper', 'bathroom stall door', 'ceiling light', 'trash bin', 'dumbbell', 'stair rail', 'tube', 'bathroom cabinet', 'cd case', 'closet rod',
    'coffee kettle', 'structure', 'shower head', 'keyboard piano', 'case of water bottles', 'coat rack', 'storage organizer', 'folded chair', 'fire alarm', 'power strip', 'calendar', 'poster', 'potted plant', 'luggage', 'mattress'
)

# Create mapping from ScanNet200 label to ID
scannet_label_to_id = {label.strip(): id for label, id in zip(CLASS_LABELS_200, VALID_CLASS_IDS_200)}

# Create case-insensitive version for matching
scannet_label_to_id_lower = {k.lower(): v for k, v in scannet_label_to_id.items()}

# Parse the mapping from 3RScan to ScanNet200
mapping_3rscan_to_scannet = {}
mapped_scannet_labels = set()
invalid_mappings = []
unmapped_3rscan_labels = []

# Process the mapping string
raw_map = '''map = {
air conditioner : machine, # generic machine category is closest match
apron : clothes, # type of clothing garment
aquarium : container, # container for holding water and fish
armchair : armchair, # exact semantic match
armoire : wardrobe, # direct semantic equivalent for large furniture piece for storage
armor : decoration, # if displayed as decorative piece
audio system : speaker, # closest match for sound equipment
baby bed : bed, # type of bed
baby changing table : table, # functional table surface
baby changing unit : table, # functional table surface
baby gym : structure, # fixed exercise equipment
baby seat : chair, # seating furniture piece
baby toys : stuffed animal, # closest match for children's playthings
backpack : backpack, # exact semantic match
bag : bag, # exact semantic match
balcony : structure, # architectural element
balcony door : door, # type of door
ball : ball, # exact semantic match
bar : bar, # exact semantic match
bar stool : stool, # type of stool
barstool : stool, # alternative spelling of bar stool
basin : sink, # bathroom sink equivalent
basket : basket, # exact semantic match
bath cabinet : bathroom cabinet, # direct equivalent
bath counter : bathroom counter, # direct equivalent
bath rack : rack, # storage framework
bath robe : clothes, # type of clothing
bathrobe : clothes, # alternative spelling of bath robe
bathroom items : object, # generic category
bathtub : bathtub, # exact semantic match
bbq : machine, # cooking equipment
beam : structure, # structural building element
bean bag : ottoman, # closest furniture match for casual seating
beanbag : ottoman, # alternative spelling of bean bag
beautician : container, # if storage unit
bed : bed, # exact semantic match
bed table : nightstand, # bedside table equivalent
bedside table : nightstand, # exact semantic match
bench : bench, # exact semantic match
beverage crate : crate, # type of crate
bicycle : bicycle, # exact semantic match
bidet : toilet, # bathroom fixture most similar
bike : bicycle, # alternative name for bicycle
bin : bin, # exact semantic match
blackboard : blackboard, # exact semantic match
blanket : blanket, # exact semantic match
blinds : blinds, # exact semantic match
board : board, # exact semantic match
body loofah : object, # generic category
boiler : object, # if portable unit, pot
book : book, # exact semantic match
books : book, # plural of book
bookshelf : bookshelf, # exact semantic match
boots : shoe, # type of footwear
bottle : bottle, # exact semantic match
bottles : bottle, # plural of bottle
bowl : bowl, # exact semantic match
box : box, # exact semantic match
boxes : box, # plural of box
bread : object, # food item
breadboard : board, # type of kitchen board
brochure : paper, # type of paper material
brush : object, # cleaning/grooming tool
bucket : bucket, # exact semantic match
buggy : cart, # wheeled transport device
bulletin board : bulletin board, # exact semantic match
cabinet : cabinet, # exact semantic match
cable : object, # electrical/utility item
cable rack : rack, # storage framework
calendar : calendar, # exact semantic match
can : container, # storage vessel
candle : candle, # exact semantic match
candles : candle, # plural of candle
candlestick : decoration, # decorative holder
canopy : structure, # overhead covering
cap : hat, # head covering
carpet : mat, # floor covering
carriage : cart, # wheeled transport device
cart : cart, # exact semantic match
case : container, # storage unit
ceiling : ceiling, # exact semantic match
ceiling /other room : ceiling, # alternative label for ceiling
ceiling light : ceiling light, # exact semantic match
chair : chair, # exact semantic match
chairs : chair, # plural of chair
chandelier : light, # lighting fixture
changing table : table, # functional surface
chest : storage container, # large storage unit
child chair : chair, # children's seating
child clothes : clothes, # children's garments
children's table : table, # children's furniture
cleaning agent : object, # cleaning supply
cleaning brush : object, # cleaning tool
cleanser : object, # cleaning supply
clock : clock, # exact semantic match
closet : closet, # exact semantic match
closet door : closet door, # exact semantic match
cloth : towel, # closest match for fabric material
clothes : clothes, # direct semantic match
clothes dryer : clothes dryer, # direct semantic match
clothes rack : rack, # semantic match as it's a structure for holding clothes
clutter : object, # generic catch-all category for mixed items
coat : jacket, # semantic match for outerwear
coffee : object, # as a raw material
coffee machine : coffee maker, # direct semantic equivalent
coffee maker : coffee maker, # direct semantic match
coffee table : coffee table, # direct semantic match
column : column, # direct semantic match
commode : dresser, # most similar as it's a tall chest of drawers
computer : computer tower, # semantic match for desktop computer
computer desk : desk, # semantic match for workspace furniture
console : tv stand, # if used for entertainment
container : container, # direct semantic match
cooking pot : object, # as a kitchen implement
corner bench : bench, # semantic match for seating
cosmetics kit : container, # as it holds items
couch : couch, # direct semantic match
couch table : end table, # semantic match for side table
counter : counter, # direct semantic match
cover : blanket, # semantic match for bed covering
cradle : bed, # as it's for sleeping
crate : crate, # direct semantic match
crib : bed, # semantic match as sleeping furniture
cube : object, # if decorative/other purpose
cup : cup, # direct semantic match
cupboard : cabinet, # semantic match for storage furniture
cups : cup, # direct semantic match (plural form)
curtain : curtain, # direct semantic match
curtain rail : shower curtain rod, # semantic match for curtain support structure
cushion : cushion, # direct semantic match
cushions stack : cushion, # semantic match (multiple cushions)
cut board : board, # semantic match for cutting surface
cutting board : board, # semantic match for cutting surface
cycling trainer : machine, # as exercise equipment
darts : decoration, # if mounted as wall art
decoration : decoration, # direct semantic match
desk : desk, # direct semantic match
desk chair : office chair, # semantic match for desk seating
device : machine, # semantic match for electronic/mechanical items
diapers : object, # as personal care item
dining chair : chair, # semantic match for dining seating
dining set : dining table, # semantic match for dining furniture set
dining table : dining table, # direct semantic match
discs : cd case, # if computer case
dish : plate, # semantic match for dining vessel
dish dryer : dish rack, # semantic match for drying dishes
dishdrainer : dish rack, # semantic match for drying dishes
dishes : plate, # semantic match (plural form)
dishwasher : dishwasher, # direct semantic match
dispenser : soap dispenser, # if for soap
documents : paper, # semantic match for written materials
dog : object, # if real/statue
doll : stuffed animal, # semantic match for toys
door : door, # direct semantic match
door /other room : door, # direct semantic match
door mat : mat, # semantic match for floor covering
doorframe : doorframe, # direct semantic match
doorframe /other room : doorframe, # direct semantic match
drain pipe : pipe, # semantic match for plumbing
drawer : cabinet, # semantic match as storage furniture component
drawers : cabinet, # semantic match as storage furniture components
drawers rack : rack, # semantic match for storage structure
dress : clothes, # as clothing item
dresser : dresser, # direct semantic match
dressing table : bathroom vanity, # if in bathroom
drinks : bottle, # if in bottles
drum : object, # as musical instrument
drying machine : clothes dryer, # semantic match for laundry appliance
drying rack : rack, # semantic match for drying structure
dumbbells : dumbbell, # direct semantic match
elevator : structure, # as building component
elliptical trainer : machine, # as exercise equipment
exhaust hood : range hood, # direct semantic match
exit sign : sign, # semantic match for informational display
extractor fan : fan, # semantic match for air movement device
fabric : object, # if raw material
fan : fan, # direct semantic match
fence : divider, # as room separator
festoon : curtain, # semantic match for decorative hanging
figure : decoration, # if ornamental
file cabinet : file cabinet, # direct semantic match
fire extinguisher : fire extinguisher, # direct semantic match
fireplace : fireplace, # direct semantic match
firewood box : storage bin, # for holding wood
flag : decoration, # if decorative
flipchart : whiteboard, # as writing surface
floor : floor, # direct semantic match
floor /other room : floor, # direct semantic match
floor lamp : lamp, # semantic match for lighting
floor mat : mat, # semantic match for floor covering
flower : plant, # semantic match for flora
flowers : plant, # semantic match for flora (plural)
flush : object, # if control mechanism
folded beach chairs : folded chair, # semantic match for portable seating
folder : container, # as storage item
folding chair : folded chair, # direct semantic match
food : object, # as consumable items
foosball table : table, # semantic match as game furniture
footstool : ottoman, # low seat for resting feet
frame : picture, # support structure for pictures/mirrors
fridge : refrigerator, # exact semantic match for cooling appliance
fruit : object, # food item
fruit plate : plate, # serving dish
fruits : object, # plural of fruit
furniture : furniture, # exact semantic match
garbage : object, # discarded material
garbage bin : trash can, # waste container
garden umbrella : furniture, # outdoor furniture
generator : machine, # power generating device
glass : cup, # drinking vessel
glass wall : structure, # transparent wall element
grass : mat, # if artificial turf
guitar : guitar, # exact semantic match
gymnastic ball : ball, # exercise ball
hair dryer : hair dryer, # exact semantic match
hand brush : object, # cleaning tool
hand dryer : hair dryer, # similar function to hair dryer
hand washer : soap dispenser, # cleaning product dispenser
handbag : purse, # carrying accessory
handhold : handicap bar, # support bar
handle : object, # apparatus for gripping
handrail : stair rail, # support rail
hanger : object, # clothes hanging device
hangers : object, # plural of hanger
hanging cabinet : cabinet, # wall-mounted storage unit
headboard : furniture, # bed component
heater : radiator, # heating device
helmet : object, # protective gear
hood : range hood, # kitchen ventilation
humidifier : machine, # air treatment device
hygiene products : object, # bathroom supplies
instrument : object, # specialized tool/device
iron : object, # clothes pressing device
ironing board : ironing board, # exact semantic match
item : object, # generic item
items : object, # plural of item
jacket : jacket, # exact semantic match
jalousie : blinds, # window covering
jar : container, # storage vessel
jug : container, # liquid container
juicer : machine, # kitchen appliance
kettle : coffee kettle, # closest match for water heating vessel
keyboard : keyboard, # exact semantic match
kids bicycle : bicycle, # children's version of bicycle
kids chair : chair, # children's version of chair
kids rocking chair : chair, # specialized children's chair
kids stool : stool, # children's version of stool
kids table : table, # children's version of table
kitchen appliance : machine, # general kitchen equipment
kitchen cabinet : kitchen cabinet, # exact semantic match
kitchen counter : kitchen counter, # exact semantic match
kitchen hood : range hood, # ventilation system
kitchen item : object, # kitchen-specific item
kitchen object : object, # kitchen-specific item
kitchen playset : object, # children's toy
kitchen rack : rack, # storage framework
kitchen sink : sink, # exact semantic match
kitchen sofa : couch, # kitchen seating
kitchen towel : towel, # kitchen-specific towel
knife box : container, # storage for utensils
ladder : ladder, # exact semantic match
lamp : lamp, # exact semantic match
laptop : laptop, # exact semantic match
laundry basket : laundry basket, # exact semantic match
letter : paper, # written communication
light : light, # exact semantic match
linen : towel, # fabric item
locker : storage container, # storage unit
lockers : storage container, # plural of locker
loft bed : bed, # elevated sleeping furniture
lounger : sofa chair, # reclining chair
luggage : luggage, # exact semantic match
machine : machine, # exact semantic match
magazine : paper, # reading material
magazine files : paper, # stored reading material
magazine rack : rack, # display framework
magazine stand : stand, # display support
mandarins : object, # food item
mannequin : decoration, # display figure
mask : decoration, # face covering
mattress : mattress, # exact semantic match
medical device : object, # if portable
menu : paper, # restaurant listing
meter : object, # measuring device
microwave : microwave, # exact semantic match
milk : object, # food item
mirror : mirror, # exact semantic match
monitor : monitor, # exact semantic match
mop : object, # cleaning tool
multicooker : machine, # cooking device
napkins : paper towel roll, # closest match for table napkins
newspaper : paper, # printed material
newspaper rack : rack, # display framework
nightstand : nightstand, # exact semantic match
notebook : book, # bound paper
notebooks : laptop, # portable computer
object : object, # exact semantic match
objects : object, # plural of object
office chair : office chair, # exact semantic match
office table : desk, # table for office work
organizer : storage organizer, # exact match if available
ottoman : ottoman, # exact semantic match
oven : oven, # exact semantic match
oven glove : object, # kitchen safety item
pack : storage container, # container for items
package : storage container, # wrapped container
packs : storage container, # plural of pack
painting : picture, # wall art
pan : object, # cooking vessel
paper : paper, # exact semantic match
paper cutter : paper cutter, # exact semantic match
paper holder : toilet paper holder, # closest match for paper holding device
paper sign : sign, # exact semantic match
paper stack : paper, # collection of paper
paper towel : paper towel roll, # exact semantic match
paper towel dispenser : paper towel dispenser, # exact semantic match
papers : paper, # plural of paper
partition : divider, # room separator
pavement : floor, # ground surface
pc : computer tower, # desktop computer
pepper : object, # food seasoning
pet bed : cushion, # animal sleeping surface
photo frame : picture, # display frame
photos : picture, # photographic images
piano : piano, # exact semantic match
picture : picture, # exact semantic match
pictures : picture, # plural of picture
pile : object, # collection of stacked items
pile of books : book, # stacked reading materials
pile of bottles : bottle, # stacked containers
pile of candles : candle, # stacked light sources
pile of folders : paper, # stacked document holders
pile of papers : paper, # stacked documents
pile of pillows : pillow, # stacked cushions
pile of wires : object, # electrical components
pillar : pillar, # exact semantic match
pillow : pillow, # exact semantic match
pin board wall : bulletin board, # wall mounting board
pipe : pipe, # exact semantic match
plank : board, # wooden board
plant : plant, # exact semantic match
planter : plant, # container for plants
plants : plant, # plural of plant
plate : plate, # exact semantic match
plates : plate, # plural of plate
platform : structure, # raised surface
player : speaker, # audio device
pocket : storage container, # small storage space
podest : structure, # raised platform
pooh : stuffed animal, # plush toy
poster : poster, # exact semantic match
pot : container, # cooking vessel
price tag : object, # item label
printer : printer, # exact semantic match
projector : projector, # exact semantic match
puf : cushion, # soft seating
puppet : stuffed animal, # closest match for toy figure
rack : rack, # exact semantic match
radiator : radiator, # exact semantic match
radio : speaker, # audio device
rag : towel, # cleaning cloth
rail : rail, # exact semantic match
railing : rail, # safety barrier
ramp : structure, # inclined surface
recycle bin : recycling bin, # exact semantic match
refrigerator : refrigerator, # exact semantic match
rocking chair : chair, # specialized chair type
roll : paper towel roll, # rolled paper product
rolled carpet : mat, # floor covering
rolling cart : cart, # wheeled transport
rolling pin : object, # kitchen tool
roof : structure, # building top
round table : table, # circular table
rowing machine : machine, # exercise equipment
rubbish bin : trash can, # waste container
rug : mat, # floor covering
sack : bag, # carrying container
salad : object, # food item
salt : object, # food seasoning
sauce boat : container, # serving vessel
scale : scale, # exact semantic match
scarf : clothes, # clothing accessory
screen : monitor, # display device
seat : seat, # exact semantic match
seat pad : cushion, # comfort padding
sewing machine : machine, # crafting device
shades : blinds, # window covering
shampoo : object, # cleaning product
sheets : blanket, # bed covering
shelf : shelf, # exact semantic match
shelf clutter : object, # miscellaneous items
shelf of caps : hat, # stored headwear
shelf unit : shelf, # storage furniture
shelves : shelf, # plural of shelf
shirt : clothes, # clothing item
shoe : shoe, # exact semantic match
shoe box : box, # footwear container
shoe commode : rack, # footwear storage
shoe rack : rack, # footwear storage frame
shoe shelf : shelf, # footwear storage surface
shoes : shoe, # plural of shoe
showcase : cabinet, # display furniture
shower : shower, # exact semantic match
shower curtain : shower curtain, # exact semantic match
shower door : shower door, # exact semantic match
shower floor : shower floor, # exact semantic match
shower wall : shower wall, # exact semantic match
side table : end table, # exact semantic match
sideboard : cabinet, # dining room storage
sidecouch : couch, # seating furniture
sidetable : end table, # alternative spelling of side table
sign : sign, # exact semantic match
sink : sink, # exact semantic match
sink counter : counter, # sink surround
slanted wall : wall, # angled wall structure
snowboard : object, # sports equipment
soap : soap dispenser, # cleaning product
soap dish : soap dish, # exact semantic match
soap dispenser : soap dispenser, # exact semantic match
socket : power outlet, # electrical connection
sofa : couch, # exact semantic match
sofa chair : sofa chair, # exact semantic match
sofa couch : couch, # alternative name for sofa
speaker : speaker, # exact semantic match
spice : object, # food seasoning
spices : object, # plural of spice
sponge : object, # cleaning tool
spots : light, # lighting fixtures
squeezer : object, # kitchen tool
stair : stairs, # single step
stairs : stairs, # exact semantic match
stand : stand, # exact semantic match
star : decoration, # decorative element
statue : decoration, # decorative sculpture
statuette : decoration, # small decorative sculpture
stepladder : ladder, # folding ladder type
stereo : speaker, # audio equipment
stereo equipment : speaker, # audio system
stick : object, # long thin item
stool : stool, # exact semantic match
storage : storage container, # general storage
storage bin : storage bin, # exact semantic match
storage box : storage container, # box for storage
storage container : storage container, # exact semantic match
storage unit : storage container, # larger storage solution
stove : stove, # exact semantic match
stroller : cart, # baby transport
stuffed animal : stuffed animal, # exact semantic match
sugar packs : object, # food sweetener
suitcase : suitcase, # exact semantic match
switch : light switch, # exact semantic match
t-shirt : clothes, # clothing item
table : table, # exact semantic match
table lamp : lamp, # exact semantic match
table soccer : table, # game table
tablet : laptop, # portable computing device
teapot : container, # beverage container
teddy bear : stuffed animal, # plush toy
telephone : telephone, # exact semantic match
tennis raquet : object, # sports equipment
tent : object, # portable shelter
things : object, # generic items
tile : object, # surface covering
tire : object, # wheel component
tissue pack : tissue box, # exact semantic match
toaster : toaster, # exact semantic match
toilet : toilet, # exact semantic match
toilet brush : object, # cleaning tool
toilet paper : toilet paper, # exact semantic match
toilet paper dispenser : toilet paper dispenser, # exact semantic match
toilet paper holder : toilet paper holder, # exact semantic match
toiletry : object, # bathroom items
tool wall : board, # mounting surface
towel : towel, # exact semantic match
towel basket : basket, # towel storage
towels : towel, # plural of towel
toy : stuffed animal, # plaything
toy house : object, # children's playset
trash bin : trash can, # exact semantic match
trash can : trash can, # exact semantic match
trashcan : trash can, # alternative spelling
tray : tray, # exact semantic match
treadmill : machine, # exercise equipment
tree : plant, # living plant
tree decoration : decoration, # ornamental item
tube : tube, # exact semantic match
tv : tv, # exact semantic match
tv stand : tv stand, # exact semantic match
tv table : tv stand, # table for television
typewriter : machine, # typing device
ukulele : guitar, # string instrument
umbrella : object, # rain protection
upholstered wall : wall, # padded wall surface
urinal : toilet, # bathroom fixture
utensils : object, # kitchen tools
vacuum : vacuum cleaner, # cleaning device
vacuum cleaner : vacuum cleaner, # exact semantic match
vase : decoration, # ornamental container
ventilation : vent, # air system
ventilator : fan, # air circulation device
wall : wall, # exact semantic match
wall /other room : wall, # alternative label for wall
wall frame : doorframe, # wall opening frame
wall plants : plant, # mounted vegetation
wall rack : rack, # wall-mounted storage
wardrobe : wardrobe, # exact semantic match
wardrobe door : closet door, # storage unit door
washbasin : sink, # bathroom fixture
washing basket : laundry basket, # clothes container
washing machine : washing machine, # exact semantic match
washing powder : object, # cleaning product
water : object, # liquid
water heater : structure, # heating system
watering can : container, # plant care tool
weights : dumbbell, # exercise equipment
weighths : dumbbell, # alternative spelling of weights
whiteboard : whiteboard, # exact semantic match
window : window, # exact semantic match
window board : windowsill, # window base
window clutter : object, # window items
window frame : doorframe, # window structure
windows : window, # plural of window
windowsill : windowsill, # exact semantic match
wood : object, # building material
wood box : box, # wooden container
xbox : computer tower, # gaming device
}'''

# Create the mapping and track issues
for line in raw_map.split('\n'):
    if ':' not in line or line.strip().startswith('#'):
        continue
        
    try:
        src, rest = line.split(':', 1)
        src = src.strip()
        target = rest.split('#')[0].strip().rstrip(',')
        
        if src and target:
            mapping_3rscan_to_scannet[src] = target
            if target.lower() in scannet_label_to_id_lower:
                mapped_scannet_labels.add(target)
            else:
                invalid_mappings.append((src, target))

    except Exception as e:
        print(f"Error processing line: {line}")
        continue

# Find unmapped ScanNet200 labels
unmapped_scannet_labels = set(CLASS_LABELS_200) - mapped_scannet_labels

# Create the final ID mapping
mapping_3rscan_to_id = {}
for src, target in mapping_3rscan_to_scannet.items():
    if target.lower() in scannet_label_to_id_lower:
        mapping_3rscan_to_id[src] = scannet_label_to_id_lower[target.lower()]

print("Invalid mappings (3RScan label -> invalid ScanNet200 label):")
for src, target in sorted(invalid_mappings):
    print(f"  {src} -> {target}")

print("\nUnmapped ScanNet200 labels:")
for label in sorted(unmapped_scannet_labels):
    print(f"  {label}")

print("\nExample of valid mappings (first 10):")
for src, id in list(mapping_3rscan_to_id.items())[:10]:
    print(f"  {src} -> {scannet_label_to_id[mapping_3rscan_to_scannet[src]]} ({mapping_3rscan_to_scannet[src]})")

# Parse 3RScan IDs from the second file
threerscan_ids = {}
with open('paste-2.txt', 'r') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) == 2:
            id_num, label = parts
            threerscan_ids[label.strip()] = int(id_num)

# Extract original comments from the mapping
original_comments = {}
for line in raw_map.split('\n'):
    if ':' not in line or line.strip().startswith('#'):
        continue
    try:
        src, rest = line.split(':', 1)
        src = src.strip()
        if '#' in rest:
            target, comment = rest.split('#', 1)
            target = target.strip().rstrip(',')
            comment = comment.strip()
            original_comments[src] = comment
    except Exception:
        continue

# Save the mapping to a file
with open('3rscan_to_scannet200_ids.txt', 'w') as f:
    f.write("# Mapping from 3RScan labels to ScanNet200 IDs\n")
    f.write("# Format: 3rscan_id: scannet200_id, # 3rscan_label --> scannet200_label: reason\n\n")
    
    for src, scannet_id in sorted(mapping_3rscan_to_id.items(), key=lambda x: threerscan_ids.get(x[0], 999999)):
        target = mapping_3rscan_to_scannet[src]
        reason = original_comments.get(src, "no reason provided")
        threerscan_id = threerscan_ids.get(src, "unknown_id")
        f.write(f"{threerscan_id}: {scannet_id}, # {src} --> {target}: {reason}\n")