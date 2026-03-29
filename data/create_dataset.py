"""
Dataset Builder — Udyam NIC Code Predictor
===========================================
Simulates real-world Udyam portal registrations.

Real people type things like:
  "I make soap at home"
  "we grow rice and wheat"
  "my shop sells mobile phones"

Each is mapped to the correct NIC 2008 sub-class code.

Data sources this replicates:
  - NIC 2008 official descriptions (MoSPI)
  - MCA Company Master Data patterns (data.gov.in)
  - Udyam Registration activity descriptions
  - GeM portal product descriptions
"""

import csv
import json
import random

random.seed(42)

# ============================================================
# CORE DATA: (plain_english_description, nic_code, nic_label)
# ============================================================
# Format: (user_typed_text, NIC_subclass_code, official_label, division)

RAW_DATA = [

    # ---- AGRICULTURE (Section A) ----
    ("i grow wheat on my farm", "1111", "Growing of wheat", "Agriculture"),
    ("wheat farming", "1111", "Growing of wheat", "Agriculture"),
    ("we cultivate wheat crops", "1111", "Growing of wheat", "Agriculture"),
    ("my business is wheat cultivation", "1111", "Growing of wheat", "Agriculture"),
    ("farmer growing wheat in punjab", "1111", "Growing of wheat", "Agriculture"),

    ("i grow rice", "1121", "Organic farming of basmati rice", "Agriculture"),
    ("basmati rice farming", "1121", "Organic farming of basmati rice", "Agriculture"),
    ("we export organic basmati rice", "1121", "Organic farming of basmati rice", "Agriculture"),
    ("paddy cultivation", "1121", "Organic farming of basmati rice", "Agriculture"),
    ("rice cultivation in irrigated fields", "1121", "Organic farming of basmati rice", "Agriculture"),
    ("i grow non basmati rice", "1122", "Organic farming of non-basmati rice", "Agriculture"),

    ("vegetable farming", "1131", "Growing of vegetables", "Agriculture"),
    ("i grow onions and tomatoes", "1132", "Growing of fruit-bearing vegetables", "Agriculture"),
    ("we grow onion", "1133", "Growing of onion", "Agriculture"),
    ("onion farming in nashik", "1133", "Growing of onion", "Agriculture"),
    ("potato farming", "1135", "Growing of potatoes and tubers", "Agriculture"),
    ("i grow potatoes", "1135", "Growing of potatoes and tubers", "Agriculture"),

    ("sugarcane farming", "1140", "Growing of sugar cane", "Agriculture"),
    ("we grow sugarcane", "1140", "Growing of sugar cane", "Agriculture"),
    ("sugarcane cultivation", "1140", "Growing of sugar cane", "Agriculture"),

    ("cotton farming", "1161", "Growing of cotton", "Agriculture"),
    ("i grow cotton", "1161", "Growing of cotton", "Agriculture"),
    ("jute farming", "1162", "Growing of jute", "Agriculture"),

    ("i grow mangoes", "1221", "Growing of mangoes", "Agriculture"),
    ("mango orchard", "1221", "Growing of mangoes", "Agriculture"),
    ("mango farming in konkan", "1221", "Growing of mangoes", "Agriculture"),
    ("banana farming", "1222", "Growing of bananas", "Agriculture"),
    ("we grow bananas", "1222", "Growing of bananas", "Agriculture"),
    ("coconut farming", "1261", "Growing of coconut", "Agriculture"),
    ("i grow coconuts", "1261", "Growing of coconut", "Agriculture"),

    ("tea plantation", "1271", "Growing of tea", "Agriculture"),
    ("we grow tea", "1271", "Growing of tea", "Agriculture"),
    ("coffee plantation", "1272", "Growing of coffee", "Agriculture"),
    ("i grow ginger", "1281", "Growing of ginger", "Agriculture"),
    ("spice farming", "1284", "Growing of spices", "Agriculture"),
    ("chili farming", "1282", "Growing of chili", "Agriculture"),

    ("i keep bees and sell honey", "1492", "Bee-keeping and production of honey", "Agriculture"),
    ("honey production", "1492", "Bee-keeping and production of honey", "Agriculture"),
    ("apiary business", "1492", "Bee-keeping and production of honey", "Agriculture"),

    ("poultry farming", "1461", "Raising and breeding of poultry", "Agriculture"),
    ("i raise chickens", "1461", "Raising and breeding of poultry", "Agriculture"),
    ("egg production farm", "1462", "Production of eggs", "Agriculture"),
    ("we produce eggs", "1462", "Production of eggs", "Agriculture"),
    ("dairy farming", "1412", "Production of milk from cows or buffaloes", "Agriculture"),
    ("i have a cow dairy", "1412", "Production of milk from cows or buffaloes", "Agriculture"),
    ("milk production", "1412", "Production of milk from cows or buffaloes", "Agriculture"),
    ("goat farming", "1441", "Raising and breeding of sheep and goats", "Agriculture"),

    ("fish farming", "3221", "Fish farming in freshwater", "Agriculture"),
    ("aquaculture freshwater", "3221", "Fish farming in freshwater", "Agriculture"),
    ("prawn farming", "3221", "Fish farming in freshwater", "Agriculture"),
    ("i do fishing in sea", "3111", "Fishing in ocean and coastal waters", "Agriculture"),

    # ---- MINING (Section B) ----
    ("coal mining", "5101", "Opencast mining of hard coal", "Mining"),
    ("we mine coal", "5102", "Belowground mining of hard coal", "Mining"),
    ("iron ore mining", "7100", "Mining of iron ores", "Mining"),
    ("granite quarrying", "8102", "Quarrying of granite", "Mining"),
    ("marble quarrying", "8101", "Quarrying of marble", "Mining"),
    ("limestone mining", "8107", "Mining of limestone", "Mining"),
    ("sand quarrying", "8106", "Operation of sand or gravel pits", "Mining"),

    # ---- FOOD MANUFACTURING (Section C) ----
    ("rice mill", "10612", "Rice milling", "Manufacturing"),
    ("i run a rice mill", "10612", "Rice milling", "Manufacturing"),
    ("rice milling unit", "10612", "Rice milling", "Manufacturing"),
    ("flour mill", "10611", "Flour milling", "Manufacturing"),
    ("we grind flour", "10611", "Flour milling", "Manufacturing"),
    ("chakki flour mill", "10611", "Flour milling", "Manufacturing"),
    ("dal mill", "10613", "Dal (pulses) milling", "Manufacturing"),

    ("we make bread and biscuits", "10712", "Manufacture of biscuits and cakes", "Manufacturing"),
    ("bakery products", "10712", "Manufacture of biscuits and cakes", "Manufacturing"),
    ("i make bread", "10711", "Manufacture of bread", "Manufacturing"),
    ("bakery unit", "10711", "Manufacture of bread", "Manufacturing"),

    ("sugar manufacturing", "10721", "Manufacture of sugar from sugarcane", "Manufacturing"),
    ("we make jaggery gur", "10722", "Manufacture of gur from sugarcane", "Manufacturing"),
    ("jaggery making", "10722", "Manufacture of gur from sugarcane", "Manufacturing"),

    ("chocolate making", "10732", "Manufacture of chocolate", "Manufacturing"),
    ("sweet shop making mithai", "10734", "Manufacture of sweetmeats", "Manufacturing"),
    ("we make indian sweets", "10734", "Manufacture of sweetmeats", "Manufacturing"),
    ("noodles and pasta manufacturing", "10740", "Manufacture of macaroni and noodles", "Manufacturing"),
    ("i make papad", "10796", "Manufacture of papads and appalam", "Manufacturing"),
    ("papad making unit", "10796", "Manufacture of papads and appalam", "Manufacturing"),

    ("tea processing", "10791", "Processing and blending of tea", "Manufacturing"),
    ("tea packing business", "10791", "Processing and blending of tea", "Manufacturing"),
    ("spice grinding and processing", "10795", "Grinding and processing of spices", "Manufacturing"),
    ("masala powder making", "10795", "Grinding and processing of spices", "Manufacturing"),
    ("we grind spices", "10795", "Grinding and processing of spices", "Manufacturing"),
    ("iodized salt processing", "10798", "Processing of salt into food grade salt", "Manufacturing"),

    ("dairy product making ghee butter", "10504", "Manufacture of cream butter cheese ghee", "Manufacturing"),
    ("i make ghee", "10504", "Manufacture of cream butter cheese ghee", "Manufacturing"),
    ("milk powder making", "10502", "Manufacture of milk powder", "Manufacturing"),
    ("ice cream manufacturing", "10505", "Manufacture of ice cream", "Manufacturing"),

    ("fish processing", "10204", "Processing and preserving of fish", "Manufacturing"),
    ("we dry and pack fish", "10201", "Sun-drying of fish", "Manufacturing"),
    ("pickle making", "10306", "Manufacture of pickles and chutney", "Manufacturing"),
    ("i make pickles", "10306", "Manufacture of pickles and chutney", "Manufacturing"),
    ("fruit juice manufacturing", "10304", "Manufacture of fruit juices", "Manufacturing"),
    ("we make fruit juice", "10304", "Manufacture of fruit juices", "Manufacturing"),
    ("jam jelly making", "10305", "Manufacture of sauces jams and jellies", "Manufacturing"),

    ("poultry meat processing", "10104", "Poultry slaughtering and preparation", "Manufacturing"),
    ("cattle feed manufacturing", "10801", "Manufacture of cattle feed", "Manufacturing"),

    # ---- BEVERAGES (Section C) ----
    ("aerated drinks manufacturing", "11041", "Manufacture of aerated drinks", "Manufacturing"),
    ("soft drink manufacturing", "11045", "Manufacture of soft drinks", "Manufacturing"),
    ("mineral water plant", "11043", "Manufacture of mineral water", "Manufacturing"),
    ("packaged water business", "11043", "Manufacture of mineral water", "Manufacturing"),

    # ---- TEXTILES (Section C) ----
    ("cotton yarn spinning", "13111", "Preparation and spinning of cotton fiber", "Manufacturing"),
    ("spinning mill cotton", "13111", "Preparation and spinning of cotton fiber", "Manufacturing"),
    ("silk weaving", "13122", "Weaving of silk fabrics", "Manufacturing"),
    ("saree weaving", "13122", "Weaving of silk fabrics", "Manufacturing"),
    ("handloom weaving", "13121", "Weaving of cotton fabrics", "Manufacturing"),
    ("we weave cotton fabric", "13121", "Weaving of cotton fabrics", "Manufacturing"),
    ("power loom", "13121", "Weaving of cotton fabrics", "Manufacturing"),
    ("jute bag making", "13942", "Manufacture of cordage or rope of jute", "Manufacturing"),
    ("carpet making", "13931", "Manufacture of carpets of cotton", "Manufacturing"),
    ("knitting hosiery unit", "13913", "Manufacture of knitted synthetic fabrics", "Manufacturing"),
    ("garment stitching", "14101", "Manufacture of textile garments", "Manufacturing"),
    ("readymade garments", "14101", "Manufacture of textile garments", "Manufacturing"),
    ("i make clothes", "14101", "Manufacture of textile garments", "Manufacturing"),
    ("tailoring unit", "14105", "Custom tailoring", "Manufacturing"),
    ("i am a tailor", "14105", "Custom tailoring", "Manufacturing"),
    ("embroidery work", "13991", "Embroidery work and making of laces", "Manufacturing"),
    ("zari work business", "13992", "Zari work and ornamental trimmings", "Manufacturing"),

    # ---- LEATHER (Section C) ----
    ("leather bag making", "15121", "Manufacture of travel goods and bags", "Manufacturing"),
    ("i make leather bags", "15121", "Manufacture of travel goods and bags", "Manufacturing"),
    ("shoe making", "15201", "Manufacture of leather footwear", "Manufacturing"),
    ("i make shoes chappal", "15201", "Manufacture of leather footwear", "Manufacturing"),
    ("footwear manufacturing", "15201", "Manufacture of leather footwear", "Manufacturing"),

    # ---- WOOD & PAPER (Section C) ----
    ("furniture making wood", "31001", "Manufacture of furniture of wood", "Manufacturing"),
    ("wooden furniture unit", "31001", "Manufacture of furniture of wood", "Manufacturing"),
    ("i make wooden furniture", "31001", "Manufacture of furniture of wood", "Manufacturing"),
    ("plywood manufacturing", "16211", "Manufacture of plywood and veneer", "Manufacturing"),
    ("paper manufacturing", "17013", "Manufacture of paper", "Manufacturing"),
    ("cardboard box making", "17023", "Manufacture of card board boxes", "Manufacturing"),
    ("paper bag making", "17024", "Manufacture of sacks and bags of paper", "Manufacturing"),

    # ---- CHEMICALS (Section C) ----
    ("soap making", "20231", "Manufacture of soap all forms", "Manufacturing"),
    ("i make soap at home", "20231", "Manufacture of soap all forms", "Manufacturing"),
    ("handmade soap business", "20231", "Manufacture of soap all forms", "Manufacturing"),
    ("detergent manufacturing", "20233", "Manufacture of detergent", "Manufacturing"),
    ("we make washing powder", "20233", "Manufacture of detergent", "Manufacturing"),
    ("paint manufacturing", "20221", "Manufacture of paints and varnishes", "Manufacturing"),
    ("fertilizer manufacturing", "20121", "Manufacture of urea and organic fertilizers", "Manufacturing"),
    ("pesticide manufacturing", "20211", "Manufacture of insecticides", "Manufacturing"),
    ("agarbatti incense sticks making", "20238", "Manufacture of agarbatti", "Manufacturing"),
    ("i make agarbatti", "20238", "Manufacture of agarbatti", "Manufacturing"),
    ("candle making", "20238", "Manufacture of agarbatti and similar products", "Manufacturing"),
    ("phenyl disinfectant making", "20212", "Manufacture of disinfectants", "Manufacturing"),
    ("hair oil shampoo manufacturing", "20236", "Manufacture of hair oil and shampoo", "Manufacturing"),
    ("cosmetics making", "20237", "Manufacture of cosmetics and toileteries", "Manufacturing"),
    ("perfume making", "20234", "Manufacture of perfumes", "Manufacturing"),
    ("toothpaste manufacturing", "20235", "Manufacture of dental hygiene preparations", "Manufacturing"),
    ("matchbox making", "20291", "Manufacture of matches", "Manufacturing"),
    ("fireworks manufacturing", "20292", "Manufacture of explosives and fireworks", "Manufacturing"),

    # ---- PHARMA (Section C) ----
    ("medicine manufacturing allopathic", "21002", "Manufacture of allopathic pharmaceutical preparations", "Manufacturing"),
    ("ayurvedic medicine making", "21003", "Manufacture of ayurvedic pharmaceutical preparations", "Manufacturing"),
    ("herbal medicine production", "21003", "Manufacture of ayurvedic pharmaceutical preparations", "Manufacturing"),
    ("homeopathic medicine", "21004", "Manufacture of homoeopathic pharmaceutical preparations", "Manufacturing"),
    ("surgical cotton bandage making", "21006", "Manufacture of medical wadding gauze bandages", "Manufacturing"),

    # ---- RUBBER & PLASTIC (Section C) ----
    ("plastic product manufacturing", "22201", "Manufacture of semi-finished plastic products", "Manufacturing"),
    ("plastic bag making", "22203", "Manufacture of plastic articles for packing", "Manufacturing"),
    ("we make plastic bags", "22203", "Manufacture of plastic articles for packing", "Manufacturing"),
    ("plastic chair table making", "22202", "Manufacture of plastic household articles", "Manufacturing"),
    ("rubber product manufacturing", "22191", "Manufacture of rubber plates and tubes", "Manufacturing"),
    ("tyre manufacturing", "22111", "Manufacture of rubber tyres for motor vehicles", "Manufacturing"),

    # ---- GLASS & CERAMICS (Section C) ----
    ("glass bangle making", "23106", "Manufacture of glass bangles", "Manufacturing"),
    ("pottery making", "23931", "Manufacture of articles of porcelain and pottery", "Manufacturing"),
    ("ceramic tile manufacturing", "23921", "Manufacture of bricks", "Manufacturing"),
    ("brick manufacturing", "23921", "Manufacture of bricks", "Manufacturing"),
    ("cement manufacturing", "23941", "Manufacture of clinkers and cement", "Manufacturing"),

    # ---- METALS (Section C) ----
    ("steel manufacturing", "24103", "Manufacture of steel", "Manufacturing"),
    ("iron and steel plant", "24101", "Manufacture of pig iron", "Manufacturing"),
    ("aluminium product making", "24202", "Manufacture of aluminium products", "Manufacturing"),
    ("copper wire making", "24201", "Manufacture of copper products", "Manufacturing"),
    ("utensil making brass copper", "25994", "Manufacture of metal household articles", "Manufacturing"),
    ("brassware manufacturing", "25994", "Manufacture of metal household articles", "Manufacturing"),
    ("lock and key making", "25934", "Manufacture of padlocks and locks", "Manufacturing"),
    ("wire fencing making", "25993", "Manufacture of metal cable and wire", "Manufacturing"),
    ("tin box can making", "25992", "Manufacture of containers tins and cans", "Manufacturing"),
    ("bolt nut manufacturing", "25991", "Manufacture of metal fasteners", "Manufacturing"),
    ("cutting tool manufacturing", "25931", "Manufacture of cutlery", "Manufacturing"),
    ("knife scissors making", "25931", "Manufacture of cutlery", "Manufacturing"),

    # ---- ELECTRONICS (Section C) ----
    ("mobile phone manufacturing", "26305", "Manufacture of pagers cellular phones", "Manufacturing"),
    ("we assemble mobile phones", "26305", "Manufacture of pagers cellular phones", "Manufacturing"),
    ("led bulb making", "27400", "Manufacture of electric lighting equipment", "Manufacturing"),
    ("electric fan manufacturing", "27503", "Manufacture of electric fans", "Manufacturing"),
    ("wire and cable manufacturing", "27320", "Manufacture of wires and cables", "Manufacturing"),
    ("switch socket manufacturing", "27331", "Manufacture of switch and socket", "Manufacturing"),
    ("battery manufacturing", "27201", "Manufacture of primary cells and batteries", "Manufacturing"),
    ("computer assembly", "26201", "Manufacture of desktop and laptop computers", "Manufacturing"),
    ("electronic component making", "26101", "Manufacture of electronic components", "Manufacturing"),
    ("pcb printed circuit board", "26104", "Manufacture of printed circuit boards", "Manufacturing"),

    # ---- AUTO (Section C) ----
    ("car manufacturing", "29101", "Manufacture of passenger cars", "Manufacturing"),
    ("two wheeler manufacturing", "30911", "Manufacture of motorcycles and scooters", "Manufacturing"),
    ("auto parts manufacturing", "29301", "Manufacture of motor vehicle parts", "Manufacturing"),
    ("tractor manufacturing", "28211", "Manufacture of tractors for agriculture", "Manufacturing"),
    ("we make electric vehicles", "29101", "Manufacture of passenger cars", "Manufacturing"),
    ("bicycle manufacturing", "30921", "Manufacture of bicycles", "Manufacturing"),

    # ---- MACHINERY (Section C) ----
    ("pump manufacturing", "28132", "Manufacture of pumps and compressors", "Manufacturing"),
    ("weighing machine manufacturing", "28194", "Manufacture of weighing machinery", "Manufacturing"),
    ("sewing machine manufacturing", "28265", "Manufacture of sewing machines", "Manufacturing"),
    ("agricultural equipment making", "28212", "Manufacture of agricultural machinery", "Manufacturing"),

    # ---- PRINTING (Section C) ----
    ("printing press", "18112", "Printing of books and magazines", "Manufacturing"),
    ("newspaper printing", "18111", "Printing of newspapers", "Manufacturing"),
    ("packaging printing", "18119", "Printing activities", "Manufacturing"),

    # ---- JEWELLERY (Section C) ----
    ("gold jewellery making", "32111", "Manufacture of jewellery of gold silver", "Manufacturing"),
    ("i make gold jewellery", "32111", "Manufacture of jewellery of gold silver", "Manufacturing"),
    ("silver jewellery", "32111", "Manufacture of jewellery of gold silver", "Manufacturing"),
    ("imitation artificial jewellery", "32120", "Manufacture of imitation jewellery", "Manufacturing"),
    ("diamond cutting and polishing", "32112", "Working of diamonds and precious stones", "Manufacturing"),

    # ---- SPORTS & TOYS (Section C) ----
    ("sports goods manufacturing", "32300", "Manufacture of sports goods", "Manufacturing"),
    ("toy making", "32401", "Manufacture of dolls and toy animals", "Manufacturing"),
    ("wooden toy making", "32401", "Manufacture of dolls and toy animals", "Manufacturing"),

    # ---- MEDICAL DEVICES (Section C) ----
    ("surgical instrument making", "32504", "Manufacture of bone plates and syringes", "Manufacturing"),
    ("medical device manufacturing", "32504", "Manufacture of medical instruments", "Manufacturing"),
    ("spectacle frame making", "22206", "Manufacture of spectacle frames of plastic", "Manufacturing"),

    # ---- CONSTRUCTION (Section F) ----
    ("building construction", "41001", "Construction of buildings", "Construction"),
    ("house construction", "41001", "Construction of buildings", "Construction"),
    ("civil construction contractor", "41001", "Construction of buildings", "Construction"),
    ("road construction", "42101", "Construction of highways and roads", "Construction"),
    ("highway building", "42101", "Construction of highways and roads", "Construction"),
    ("bridge construction", "42101", "Construction of bridges and tunnels", "Construction"),
    ("electrical installation", "43211", "Installation of electrical wiring", "Construction"),
    ("plumbing work", "43221", "Installation of plumbing", "Construction"),
    ("painting contractor", "43303", "Interior and exterior painting", "Construction"),
    ("interior design work", "43303", "Interior painting and decorating", "Construction"),
    ("real estate developer", "41001", "Construction of buildings", "Construction"),

    # ---- WHOLESALE & RETAIL (Section G) ----
    ("grocery shop", "47211", "Retail sale of cereals and pulses", "Trade"),
    ("kirana store", "47211", "Retail sale of food products", "Trade"),
    ("general store", "47190", "Other retail sale in non-specialized stores", "Trade"),
    ("mobile phone shop", "47414", "Retail sale of telecommunication equipment", "Trade"),
    ("electronic goods shop", "47420", "Retail sale of audio and video equipment", "Trade"),
    ("clothing shop", "47711", "Retail sale of readymade garments", "Trade"),
    ("medicine pharmacy", "47721", "Retail sale of pharmaceuticals", "Trade"),
    ("jewellery shop", "47733", "Retail sale of jewellery", "Trade"),
    ("petrol pump", "47300", "Retail sale of automotive fuel", "Trade"),
    ("book shop stationery", "47613", "Retail sale of stationery", "Trade"),
    ("hardware shop", "47522", "Retail sale of hardware", "Trade"),
    ("vegetable fruit shop", "47212", "Retail sale of fresh fruit and vegetables", "Trade"),
    ("furniture shop", "47591", "Retail sale of household furniture", "Trade"),
    ("wholesale trader", "46209", "Wholesale of agriculture raw materials", "Trade"),
    ("online selling ecommerce", "47912", "Retail sale via e-commerce", "Trade"),

    # ---- TRANSPORT (Section H) ----
    ("truck transport", "49231", "Motorised road freight transport", "Transport"),
    ("trucking business", "49231", "Motorised road freight transport", "Transport"),
    ("taxi cab service", "49224", "Taxi operation", "Transport"),
    ("auto rickshaw", "49224", "Taxi operation", "Transport"),
    ("bus transport service", "49221", "Long distance bus services", "Transport"),
    ("courier service", "53200", "Courier activities", "Transport"),
    ("parcel delivery", "53200", "Courier activities", "Transport"),
    ("cold storage warehouse", "52101", "Warehousing of refrigerated goods", "Transport"),
    ("warehouse storage", "52102", "Warehousing non-refrigerated", "Transport"),
    ("shipping agent", "52292", "Activities of shipping cargo agents", "Transport"),
    ("travel agency", "52291", "Activities of travel agents", "Transport"),
    ("cargo handling", "52241", "Cargo handling for land transport", "Transport"),

    # ---- HOSPITALITY (Section I) ----
    ("hotel restaurant", "55101", "Hotels and motels providing lodging", "Hospitality"),
    ("dhaba food stall", "56101", "Restaurants", "Hospitality"),
    ("tiffin catering service", "56291", "Activities of food service contractors", "Hospitality"),
    ("canteen mess", "56292", "Operation of canteens", "Hospitality"),
    ("juice shop", "56303", "Fruit juice bars", "Hospitality"),
    ("tea stall", "56302", "Tea and coffee shops", "Hospitality"),
    ("fast food centre", "56102", "Cafeterias and fast food restaurants", "Hospitality"),
    ("event catering", "56210", "Event catering", "Hospitality"),

    # ---- IT & TELECOM (Section J) ----
    ("software development", "62011", "Writing and modifying computer programs", "IT Services"),
    ("web design", "62012", "Web page designing", "IT Services"),
    ("mobile app development", "62011", "Writing computer programs", "IT Services"),
    ("it consulting", "62020", "Computer consultancy", "IT Services"),
    ("data entry work", "63114", "Providing data entry services", "IT Services"),
    ("cyber cafe", "63992", "Activities of cyber cafe", "IT Services"),
    ("call centre bpo", "82200", "Activities of call centres", "IT Services"),
    ("internet service provider", "61104", "Activities providing internet access", "IT Services"),
    ("cable tv operator", "61103", "Activities of cable operators", "IT Services"),
    ("mobile recharge shop", "61201", "Activities of internet access wireless", "IT Services"),

    # ---- FINANCE (Section K) ----
    ("money lending", "64920", "Other credit granting", "Finance"),
    ("chit fund", "64910", "Financial leasing", "Finance"),
    ("insurance agent", "66220", "Activities of insurance agents and brokers", "Finance"),
    ("stock broker", "66120", "Security and commodity contracts brokerage", "Finance"),
    ("mutual fund agent", "66301", "Management of mutual funds", "Finance"),
    ("tax consultant", "69202", "Tax consultancy", "Finance"),
    ("accountant ca firm", "69201", "Accounting and auditing activities", "Finance"),

    # ---- REAL ESTATE (Section L) ----
    ("property dealer", "68200", "Real estate activities on fee basis", "Real Estate"),
    ("real estate agent broker", "68200", "Real estate activities on fee basis", "Real Estate"),

    # ---- PROFESSIONAL SERVICES (Section M) ----
    ("advocate lawyer", "69100", "Legal activities", "Professional"),
    ("architect firm", "71100", "Architectural and engineering activities", "Professional"),
    ("civil engineer consultant", "71100", "Architectural and engineering activities", "Professional"),
    ("advertising agency", "73100", "Advertising", "Professional"),
    ("photography studio", "74201", "Commercial photograph production", "Professional"),
    ("graphic design", "74103", "Services of graphic designers", "Professional"),
    ("fashion designer", "74101", "Fashion design", "Professional"),
    ("veterinary doctor", "75000", "Veterinary activities", "Professional"),
    ("market research", "73200", "Market research and public opinion polling", "Professional"),

    # ---- EDUCATION (Section P) ----
    ("coaching class tuition", "85491", "Academic tutoring services", "Education"),
    ("private school", "85211", "General school education first stage", "Education"),
    ("college", "85301", "Higher education", "Education"),
    ("skill training institute iti", "85221", "Technical and vocational education", "Education"),
    ("driving school", "85223", "Professional motor driving school", "Education"),
    ("computer training centre", "85491", "Academic tutoring services", "Education"),
    ("dance music class", "85420", "Cultural education", "Education"),

    # ---- HEALTHCARE (Section Q) ----
    ("hospital clinic", "86100", "Hospital activities", "Healthcare"),
    ("doctor medical practice", "86201", "Medical practice activities", "Healthcare"),
    ("dentist", "86202", "Dental practice activities", "Healthcare"),
    ("ayurvedic clinic", "86901", "Activities of Ayurveda practitioners", "Healthcare"),
    ("homeopathy clinic", "86903", "Activities of homeopaths", "Healthcare"),
    ("diagnostic lab pathology", "86905", "Activities of diagnostic laboratories", "Healthcare"),
    ("pharmacy medical shop", "47721", "Retail sale of pharmaceuticals", "Healthcare"),
    ("nursing home", "86100", "Hospital activities", "Healthcare"),

    # ---- PERSONAL SERVICES (Section S) ----
    ("beauty parlour salon", "96020", "Hairdressing and other beauty treatment", "Services"),
    ("dry cleaning laundry", "96010", "Washing and dry cleaning of textiles", "Services"),
    ("repair shop electronics", "95210", "Repair of consumer electronics", "Services"),
    ("mobile phone repair", "95120", "Repair of communication equipment", "Services"),
    ("bicycle repair", "95291", "Repair of bicycles", "Services"),
    ("tailoring alteration", "95292", "Repair and alteration of clothing", "Services"),
    ("cobbler shoe repair", "95230", "Repair of footwear and leather goods", "Services"),
    ("pest control", "81299", "Other building and industrial cleaning", "Services"),
    ("security guard agency", "80100", "Private security activities", "Services"),
    ("cleaning services", "81210", "General cleaning of buildings", "Services"),
    ("event management", "82300", "Organization of conventions and trade shows", "Services"),
    ("placement agency hr", "78100", "Activities of employment placement agencies", "Services"),
    ("photocopying shop", "82191", "Photocopying and duplicating services", "Services"),

    # ---- ENERGY (Section D) ----
    ("solar energy plant", "35105", "Electric power generation using solar energy", "Energy"),
    ("wind energy farm", "35106", "Electric power generation non conventional", "Energy"),
    ("electricity distribution", "35109", "Collection and distribution of electric energy", "Energy"),
    ("gas agency cylinder", "35202", "Distribution and sale of gaseous fuels", "Energy"),

    # ---- REPAIR & MAINTENANCE (Section C) ----
    ("machine repair workshop", "33121", "Repair of engines and turbines", "Manufacturing"),
    ("vehicle repair garage", "45200", "Maintenance and repair of motor vehicles", "Trade"),
    ("mobile bike repair shop", "45403", "Maintenance of motorcycles and scooters", "Trade"),
]


# ============================================================
# AUGMENTATION — Real Udyam users type many ways
# ============================================================

PREFIXES = [
    "", "", "",  # no prefix most common
    "we are into ",
    "our business is ",
    "i am doing ",
    "we deal in ",
    "this unit is involved in ",
    "the company is engaged in ",
    "business activity: ",
    "industry: ",
    "my work is ",
    "i run a ",
    "we have a ",
    "i own a ",
    "i do ",
    "we do ",
]

SUFFIXES = [
    "", "", "",  # no suffix most common
    " in india",
    " for domestic market",
    " for export",
    " on small scale",
    " as msme",
    " at home",
    " in my village",
    " in my city",
    " for local market",
]

def augment(text, nic, label, division):
    results = []
    seen = set()
    # original
    results.append((text.strip(), nic, label, division))
    seen.add(text.strip().lower())

    for _ in range(6):
        p = random.choice(PREFIXES)
        s = random.choice(SUFFIXES)
        new_text = (p + text + s).strip()
        if new_text.lower() not in seen:
            results.append((new_text, nic, label, division))
            seen.add(new_text.lower())

    return results


# ============================================================
# BUILD FINAL DATASET
# ============================================================

all_rows = []
for (text, nic, label, division) in RAW_DATA:
    all_rows.extend(augment(text, nic, label, division))

random.shuffle(all_rows)

# ============================================================
# SAVE FILES
# ============================================================

# CSV for training
csv_path = "industries.csv"
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["text", "nic_code", "nic_label", "division"])
    writer.writeheader()
    for (text, nic, label, division) in all_rows:
        writer.writerow({
            "text": text,
            "nic_code": nic,
            "nic_label": label,
            "division": division
        })

# JSON for reference
json_path = "industries.json"
json_data = [
    {"text": t, "nic_code": n, "nic_label": l, "division": d}
    for (t, n, l, d) in all_rows
]
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(json_data, f, indent=2, ensure_ascii=False)

# NIC reference lookup
nic_lookup = {}
for (_, nic, label, division) in RAW_DATA:
    nic_lookup[nic] = {"label": label, "division": division}

with open("nic_lookup.json", "w") as f:
    json.dump(nic_lookup, f, indent=2, ensure_ascii=False)

# Stats
from collections import Counter
div_counts = Counter(d for _, _, _, d in all_rows)

print("=" * 55)
print("  DATASET CREATION COMPLETE")
print("=" * 55)
print(f"  Total samples     : {len(all_rows)}")
print(f"  Unique NIC codes  : {len(set(n for _,n,_,_ in all_rows))}")
print(f"  Saved to          : {csv_path}")
print()
print("  Samples per sector:")
for div, cnt in sorted(div_counts.items(), key=lambda x: -x[1]):
    print(f"    {div:<25} {cnt}")
print("=" * 55)
