from NJ_tree_analysis_functions import start_gui_explorer

objs = [
    160108002001032, 160108002001056, 160108002001167, 160108002001169, 160108002001240, 160108002001265,
    160108002001349
]
objs = [str(o) for o in objs]
start_gui_explorer(objs,
                   manual=True,
                   kinematics_source='ucac5')