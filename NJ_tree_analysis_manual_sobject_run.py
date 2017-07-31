from NJ_tree_analysis_functions import start_gui_explorer

objs = [140305003201207,140305003201326,140307003101052,140314005201379,140608003101390,150602002701237]

objs = [str(o) for o in objs]
start_gui_explorer(objs,
                   manual=True,
                   kinematics_source='ucac5')