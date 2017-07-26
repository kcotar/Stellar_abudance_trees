from NJ_tree_analysis_functions import start_gui_explorer

objs = [140607001401109,140707003601189,140713004001062,140806002301207,140809002601086,141101001801338,141102002401260,141104002301346,150827002901074,150827002901117,150827002901121,150827002901225,150827002901270]

objs = [str(o) for o in objs]
start_gui_explorer(objs,
                   manual=True,
                   save_dir='',
                   i_seq=1,
                   kinematics_source='ucac5')