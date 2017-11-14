from NJ_tree_analysis_functions import start_gui_explorer

objs = [
    140305003201095, 140305003201103, 140305003201185, 140307002601128, 140307002601147, 140311006101253,
    140314005201008, 140608002501266, 150211004701104, 150428002601118, 150703002101192
]
objs = [
    170828002701137, 170828002701148, 170828002701164, 170828002701168, 170828002701169, 170828002701170,
    170828002701204, 170828002701211, 170828002701216, 170828002701233
]

objs = [str(o) for o in objs]
start_gui_explorer(objs,
                   manual=True, initial_only=False, loose=True,
                   kinematics_source='ucac5')
# start_gui_explorer(objs,
#                    manual=False, initial_only=False, loose=True,
#                    kinematics_source='ucac5')