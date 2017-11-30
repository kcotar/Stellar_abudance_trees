from NJ_tree_analysis_functions import start_gui_explorer

# nov v omegaCen?
objs = [
    140305003201095, 140305003201103, 140305003201185, 140307002601128, 140307002601147, 140311006101253,
    140314005201008, 140608002501266, 150211004701104, 150428002601118, 150703002101192
]

# nov v NGC6774
objs = [
    140707002601170, 140707002601363, 140806003501357, 151009001601071, 160522005601187, 170506006401032,
    170506006401321, 170506006401334, 170506006401352, 170506006401367, 170506006401373, 170506006401374,
    170506006401392, 170802004301085, 170906002601139, 170907002601241, 140708005301211,150703005601230,161013001601131,161109002601048,170506005401371,170506006401241,170506006401303,170907003101232,170907003101274,170910003101093,170506006401009, 170506006401032, 170506006401039, 170506006401063, 170506006401095, 170506006401189, 170506006401265, 170506006401281, 170506006401321, 170506006401331, 170506006401334, 170506006401345, 170506006401352, 170506006401367, 170506006401373, 170506006401374, 170506006401392
]

objs = [
150101002901201,160330001601028,161104004801242,170103002301063,170103002301071,170131001801020,170131001801188,170131001801228,170131001801279,170220001601163,170220001601199,160106002601106,160330001601038,160330001601142,160811004601237,161104004801252,161104004801264,161106005101029,161106005101180,161106005101201,161228002001053,161228002001120,170105003601015,170105003601177,170105003601221,170117002101103,170119002601214,170119002601365,170122002601114,170122002601182,170122002601183,170128002101317,170131001801118,170205002801040,170205002801209,170205002801307,170205002801385,170220001601202,170220001601263,170220001601289,170220001601356,
150209002201091,150209002201278,150830006601108,151219003101190,160110002101186,160111002101145,160330001601225,160811004601031,160811004601276,161013005401134,161013005401214,161228002001173,170105003601191,170106003601236,170106004101164,170122002601195,170122002601252,170122002601325,170122002601338,170122002601343,170122002601361,170122002601397,170131001801234,170205002801288,150209002201278,170122002601338,170122002601343,170122002601361,170122002601397,170205002801288,
150830006601042,151110004201025,161104004801038,161106005101008,161106005101031,161107004401223,161107004401364,170117002101056,170122002601011,170122002601026,170122002601054,170122002601056,170131001801148,170131001801224,170205002801024,170205002801122,
141103004201302,141104004301357,141104004801336,151110004201355,151111002601241,151111002601325,151111002601385,151220001601011,151231002601385,160330001601224,160330001601315,160330001601367,160330001601382,160330001601385,160811004601388,161013004401298,161013005401257,161013005401275,161013005401326,161013005401369,161013005401379,161106005101307,161228002001215,170105003601012,170105003601294,170122002601301,170122002601359,170122002601369,170122002601373,170122002601376,170122002601377
]


objs = [str(o) for o in objs]
start_gui_explorer(objs,
                   manual=True, initial_only=False, loose=True,
                   kinematics_source='ucac5')
# start_gui_explorer(objs,
#                    manual=False, initial_only=False, loose=True,
#                    kinematics_source='ucac5')