from src import renderer
from party import maths, pnodes
from src.party.maths import scurve
from src.party.pnodes import eval_prompt, PGlobals, PList, PProp

shape = PList('''
spirals
diagonals
crosshatching
Horizontal lines in a spiral
lines extending out radially
concentric circles in a bullseye shape
revolving barycentric circles
interlocking uneven triangles
human faces
human eyes
butterfly
flower
pentagonal polygon
sunflower made of geometric shapes
repeating concentric ripples
hexagon
sphere
blob
fractal
circle
square
brain
helix
''')

artist = PList('''
alexander john whiter
ronald balfour
barkler clive
Arcimboldo Giuseppe
Beardsley Aubrey
Barnes Ernie
''')

# Depth of field effect blur
# pixelart with bayer dither
textures = PList('''
fractal fern leaves pattern
tessellated skin from a black snake texture
green geometric stripes on a butterfly's wings
organic patterns on a butterfly's wings
bismuth crystal
fractals of beautiful fluffy blue flowers with large petals in a concentric shape
octopus Suction Cups
fractal Cells under an Electron Microscope
dark black wood
peacock pattern
texture of the French Angelfish, yellow stripes on black, inhabit reefs and sandy areas in Tropical West Atlantic; picture taken Grand Cayman.
pointillism pattern millions of small dots
bioluminescence
light reflecting on the floor passing through trees
magical bioluminescent black sand with a lot of glittery lights in the middle of the black sand specks. A bit like cities seen from space at night. It appears magical and glowy, bioluminescent. It's like a night sky with a lot of glowy colorful stars, but not galaxy.
water caustics
wavering water surface reflection
earth seen from space at night
inversion of reality mirror
time in Interlocking Gears
aurora borealis
closeup colorful parrot wings design
Rendered in Unreal Engine 4
sparkly rubies
deep at Sea
bioluminescence
''')

everything = PList('''
<motion> into motion
<motion> <dir> a slope with motion and energy
in <energy>
into <intensity> <intensity> motions
<motion> into light hypermotion energy
<motion> <align> to the horizon
<motion> into an harmonic rotation
<detail> <motion> energic
<motion> into a psychedelic vapor of arcane energies
<motion> into dmt entities
''')

colors = PList('''
There is an <color> sheen on everything...
The <color> colors are being refracted everywhere from the edges!
<color> and <color> colors are everywhere like fireflies!
<color> hues are appearing in front like holograms...
<global> is <motion> into a mix of <color> and <color> goo!
It's <motion> into a shiny black rubbery <color> hue!
<color> <color> hues are appearing next to everything!
There is a shiny <color>/<color> holographic hue coming from all sides...
The <area> are lit up by a shiny <color> and <color>!
The <area> are all reflecting with purple and <color> colors... interesting
colors in the style of technicolor <color>, <color>, <color>, and <color>
''')

chromatic = PList('basil green clover light emerald shallow sea green dark cyan blue topaz dark azure delphinium blue light blue dark lavender violet purple daisy dark magenta shocking pink dragon fruit red orange yellow ochre olive oil wasabi green grape celery')
intensity = PList("calm lively energetic frantic hypermotion arcane")
scenic = PList("scenic wide-shot cinematic beautiful vast empty clear overhead distant subtle wide-angle")
scene = PList("meadow valley river mountain forest lake ocean beach desert reef canyon cave")
# Vague composition landmarks
thing = PList("slope valley corner surface edge horizon object background")
layer = PList("background foreground middle ground sky horizon")
through = PList("through into from above below across around over under")
# A wild variety of exotic flowers
flower = PList("dahlia rose tulip daisy orchid lily sunflower poppy iris carnation marigold aster begonia geranium")
beauty = PList("beautiful jaw-dropping stunning gorgeous breathtaking awe-inspiring awe-inspiring")
detail = PList("detailed complex elaborate complicated")
motion = PList("stretching twisting flowing sliding rotating deforming morphing transforming streaking melting condensing evaporating marbling swirling")
energy = PList("energy movement motion light antimatter vapor fog")
# modifies a movement, e.g. "condensing into <align>"
align = PList("perpendicular parallel horizontal vertical")
area = PList("edges sides corners center middle rim top")
dir = PList("up down left right behind ahead")
fog = PList("fog mist vapor haze clouds foggy cotton")
palette = PList("warm,cinematic,pastel,vibrant,exotic")
plant = PList("golden pothos pointsettia cactus succulent orchid ferns ivy bamboo palm foliage leaves grasses mosses ferns vines flowers floral canopy tree shrub bush bushes wood")
house_room = PList("office bedroom kitchen living room bathroom")
nature_places = PList("jungle forest waterfall river lake pond ocean sea beach")
city_places = PList("city street alley alleyway sidewalk")
time = PList("day night dawn dusk sunset sunrise evening morning")
weather = PList("rainy stormy snowy windy cloudy foggy sunny snowy")
place = PList(""" <nature_places:s5> aesthetic with <keyword>; <coverart> with <keyword>""")
coverart = PList("""acrylic 70s prog rock album cover; artistic 90s idm album cover; acrylic 70s magazine cover art; vaporwave native american album cover;""")
artist = PList("""johannes vermeer; dslr photo; dramatic lighting; yayoi kusama; syd mead; botero; amy judd; robert mcginnis; kadir nelson""")  # audrey kawasaki
keyword = PList("""
flowers; lavender flowers; coral ornaments; autumn leaves; tree bark texture; huge tree of life; vitruvian man aesthetic;
illuminati aesthetic; depth of field; triangle; potpourri; vogue magazine aesthetic; brushery; clouds; orchid flowers;
fern leaves; glowing; made of broken glass; broken mirror;
""")
magic = PList("whimsical magical cursed haunted horrific fantastical surreal")
bgcolor = PList("black dark candlelit obsidian coal nighttime twilight charcoal midnight ink shadow raven jet ebony onyx soot tar pitch")
root = PGlobals(
        # bloom=ProportionSet(bloom, 1, 0.5, 0, 1, scale=5),
        # chromatic=SequenceSet(chromatic, scale=1.85),
        # scenes=ProportionSet(scenes, 100, [0.5, 0.6], [0.1, 0.12], 0.75, scale=4),
        # everything=ProportionSet(everything, 1, [0.1, 0.5], 0, 1, scale=2.8),

        # colors1=ProportionSet(colors1, 7.5, 0.5, (0, 1), 2.5),

        # colors2 = ProportionSet(colors2, 10, 0.5,  0, 1)
        # objects=ProportionSet(objects, 10, 0.125, [0.3, 0.25], 0.25, scale=1.5),
        # rendering = ProportionSet(rendering,  7.5, [0.95, 0.985],  [0.01, 0.25], 1),
        # ["5.75*+cos1(t,0.5,5)", objects],
        # ["cos1(t,1+cos1(t,.5,.5),0.85)", d2],
)

s1 = PProp(shape, 9, (0.5, 0.6), (0.1, 0.22), 0.5, scale=1.85)
s2 = PProp(shape, 6, (0.5, 0.6), (0.1, 0.22), 0.5, scale=1.85)
s3 = PProp(artist, 5, (0.9, 0.985), (0.01, 0.1), 1, scale=1)
s4 = PProp(textures, width=1.5, p=(0.4, 0.25), drift=.7, lerp=1, scale=1)
s5 = PProp(fog, width=0.5, p=0.1, drift=0.3, lerp=1, curve=scurve, scale=2)
confmap = {
    'scenic'    : s5,
    'texture'   : [s5, s4],
    'artist'    : [s5, s4, s3],
    'shape'     : [s5, s4],
    'beauty'    : [s5, s4, s3],
    'scene'     : [s5],
    'colors'    : [s5, s4, s3, s2],
    'intensity' : [s5, s4, s3, s2],
    'through'   : [s5, s4, s3],
    'layer'     : [s5, s4, s3],
    'flower'    : [s3, s2],
    'motion'    : [s3],
    '*'         : [s5],
    # ----------------------------------------
    'n_gem'     : s5,
    'a_gem'     : s5,
    'a_gem2'    : s5,
    'a_cave'    : s4,
    'a_shape'   : s5,
    'a_light'   : s3,
    'a_penumbra': s2,
}

n_gem = PList("chrome aluminum diamond ruby emerald sapphire quartz amethyst jade opal pearl peridot onyx turquoise agate tourmaline garnet topaz coral lapis lazuli malachite hematite pearl")
a_gem = PList("Gelatinous Lustrous Shiny Glittering Sparkling Glossy Polished Radiant Dazzling Gleaming Glossy Vibrant Incandescent Translucent Opaque Semitransparent Transparent Diaphanous Filmy Gossamer Sheer Velvety Satiny Smooth Silky Slick Glossy Brilliant Flashing Flickering Glinting Glistening Shimmering Twinkling Flashing Glistening Iridescent Luminous Opalescent Pearlescent Scintillating")
a_gem2 = PList("chrome sparkling crystalline lustrous shimmering radiant radiant dazzling dazzling glossy polished lustrous iridescent multicolored opalescent resplendent gleaming beaming luminous luminous effulgent resplendent gleaming beaming luminous luminous effulgent")
a_cave = PList("dark shadowy eerie gloomy mysterious cavernous underground subterranean dank dreary desolate forsaken spooky abandoned haunted hidden cryptic cursed forbidden uncharted eerie fathomless abyssal bottomless shadowy ghostly haunted foreboding fearful perilous treacherous")
a_shape = PList("fluorescent radiant glowing vibrant iridescent shimmering sparkling radiant auroral neon luminescent phosphorescent incandescent pulsating throbbing kaleidoscopic hypnotic hypnotizing ")
a_light = PList("glowing shimmering diffractive refractive iridescent refracted prismatic rainbow holographic holographic multicolored sparkling dazzling dazzling luminous radiant radiant effulgent beaming beaming radiant radiant effulgent")
a_penumbra = PList("shadow darkness twilight dimness obscurity shade gloom murkiness dusk duskiness gloaming nightfall eclipse lunar eclipse solar eclipse solar occultation umbra penumbra crepuscule gloaming nightfall eclipse lunar eclipse solar eclipse solar occultation umbra penumbra crepuscule")

# Nomenclature
# nprompt: node prompt (with <tags>)
# nnprompt: node root
# nprompt_last: last node prompt
# prompt: resolved prompt for this frame, to use in img2img



# artist = PromptWo
# rds("Roger Dean,Hipgnosis,Storm Thorgerson,George Hardie,David Anstey,John Billings,Barney Bubbles,Roger Huyssen,Salvador Dali,Klarwein")
# scene = "looking down an intricate mc escher optical illusion inside of a <a_cave> cave tunnel"
# scene2 = "a tunnel made out of human hands in negative space"
# style = f"70s Prog rock album cover (mint conditions) depicting {scene2}, with risograph (print by <artist>), artistic, light pastel colors, technicolor, in the style of (<artist>) cover art"
# p = f"{scene} with black <a_penumbra> distance fog. Everything is ultra detailed, has 3D overlapping depth effect, into the center, painted with neon reflective/metallic/glowing ink, covered in <a_gem> <a_gem2> gemstone / <n_gem> ornaments, <a_light> light diffraction, (vibrant and colorful colors), {style}, painted with (acrylic)"

# artist = PromptWords("Roger Dean,Hipgnosis,Storm Thorgerson,George Hardie,David Anstey,John Billings,Barney Bubbles,Roger Huyssen,Salvador Dali,Klarwein")
# scene = "looking down an intricate mcescher <a_penumbra> <a_cave> cave tunnel as an optical illusion"
# style = f"70s Prog rock album cover, a dark endless tunnel, with risograph print by <artist>, artistic, light pastel colors, technicolor, in the style of (<artist>) album art."
# p = f"{scene} Everything is ultra detailed, <a_light> light diffraction, (vibrant and colorful colors), {style}, painted with neon reflective/metallic/glowing ink, everything is overlapping, into the center, covered in <a_gem> <a_gem2> gemstone / <n_gem> ornaments, ainted with (acrylic), painted with neon reflective/metallic/glowing ink, everything is overlapping, into the center, covered in <a_gem> <a_gem2> gemstone / <n_gem> ornaments, "

# for n in pnodes.all_prompt_nodes:
#     n.print()

# d = 5
# res = 30
# for i in range(1, res):
#     t = (i / res) * d
#     # s = seq.eval_text(t)
#     s = eval_prompt(p, t)
#     # clamp s length to 60 chars max
#     m = 120
#     if len(s) > m:
#         s = s[:m] + "..."
#     print(f'{t:.02f}', s)
