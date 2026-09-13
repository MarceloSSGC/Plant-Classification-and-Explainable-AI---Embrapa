
# def config_function(config):

#     #------------------------------------------------------------------------
#     # 1. MULTIVIEW_DATA_NAME

#     ALIGN_DATA_NAME = config["ALIGN_DATA_NAME"]

#     SEG_DATA_NICKNAME = config["SEG_DATA_NICKNAME"]
#     SEG_DATA_NAME = f"{ALIGN_DATA_NAME}__{SEG_DATA_NICKNAME}"

#     MULTIVIEW_DATA_NICKNAME = config["MULTIVIEW_DATA_NICKNAME"]
#     MULTIVIEW_DATA_NAME = f"{SEG_DATA_NAME}__{MULTIVIEW_DATA_NICKNAME}"

#     config["MULTIVIEW_DATA_NAME"] = MULTIVIEW_DATA_NAME

#     #------------------------------------------------------------------------
#     # 2. SPLIT_DATA_NAME

#     SEED = config["SEED"]

#     ALIGN_DATA_NAME = config["ALIGN_DATA_NAME"]

#     SEG_DATA_NICKNAME = config["SEG_DATA_NICKNAME"]
#     SEG_DATA_NAME = f"{ALIGN_DATA_NAME}__{SEG_DATA_NICKNAME}"

#     MULTIVIEW_DATA_NICKNAME = config["MULTIVIEW_DATA_NICKNAME"]
#     MULTIVIEW_DATA_NAME = f"{SEG_DATA_NAME}__{MULTIVIEW_DATA_NICKNAME}"
#     config["MULTIVIEW_DATA_NAME"] = MULTIVIEW_DATA_NAME

#     TRAIN_SIZE = config["TRAIN_SIZE"]
#     VAL_SIZE = config["VAL_SIZE"]

#     SPLIT_DATA_NAME = config["MULTIVIEW_DATA_NAME"] + f"__SEED_{SEED}__T_{TRAIN_SIZE}_V_{VAL_SIZE}"

#     config["SPLIT_DATA_NAME"] = SPLIT_DATA_NAME

#     #------------------------------------------------------------------------
#     # 3. SPLIT_DATE_TYPE

#     MULTIVIEW_DATA_TYPE = config["MULTIVIEW_DATA_TYPE"]

#     SPLIT_DATE_TYPE = config["MULTIVIEW_DATA_TYPE"]

#     config["SPLIT_DATE_TYPE"] = SPLIT_DATE_TYPE

#     #------------------------------------------------------------------------
#     # 4. AUG_DATE_NAME

#     ALIGN_DATA_NAME = config["ALIGN_DATA_NAME"]

#     SEG_DATA_NICKNAME = config["SEG_DATA_NICKNAME"]
#     SEG_DATA_NAME = f"{ALIGN_DATA_NAME}__{SEG_DATA_NICKNAME}"

#     MULTIVIEW_DATA_NICKNAME = config["MULTIVIEW_DATA_NICKNAME"]
#     MULTIVIEW_DATA_NAME = f"{SEG_DATA_NAME}__{MULTIVIEW_DATA_NICKNAME}"
#     config["MULTIVIEW_DATA_NAME"] = MULTIVIEW_DATA_NAME

#     TRAIN_SIZE = config["TRAIN_SIZE"]
#     VAL_SIZE = config["VAL_SIZE"]

#     SPLIT_DATA_NAME = config["MULTIVIEW_DATA_NAME"] + f"__SEED_{SEED}__T_{TRAIN_SIZE}_V_{VAL_SIZE}"
#     config["SPLIT_DATA_NAME"] = SPLIT_DATA_NAME

#     AUG_DATE_NAME = SPLIT_DATA_NAME + "__AUG"

#     config["AUG_DATE_NAME"] = AUG_DATE_NAME

#     #------------------------------------------------------------------------
#     # 5. AUG_DATE_TYPE

#     MULTIVIEW_DATA_TYPE = config["MULTIVIEW_DATA_TYPE"]

#     SPLIT_DATE_TYPE = config["MULTIVIEW_DATA_TYPE"]
#     config["SPLIT_DATE_TYPE"] = SPLIT_DATE_TYPE

#     AUG_DATE_TYPE = SPLIT_DATE_TYPE + "__AUG"

#     config["AUG_DATE_TYPE"] = AUG_DATE_TYPE


#======================================================================
#======================================================================


def config_function(config):

    #------------------------------------------------------------------------
    # Base parameters

    ALIGN_DATA_NAME = config["ALIGN_DATA_NAME"]
    SEG_DATA_NICKNAME = config["SEG_DATA_NICKNAME"]
    MULTIVIEW_DATA_NICKNAME = config["MULTIVIEW_DATA_NICKNAME"]

    SEED = config["SEED"]
    TRAIN_SIZE = config["TRAIN_SIZE"]
    VAL_SIZE = config["VAL_SIZE"]

    MULTIVIEW_DATA_TYPE = config["MULTIVIEW_DATA_TYPE"]

    #------------------------------------------------------------------------
    # MULTIVIEW

    SEG_DATA_NAME = f"{ALIGN_DATA_NAME}__{SEG_DATA_NICKNAME}"

    MULTIVIEW_DATA_NAME = f"{SEG_DATA_NAME}__{MULTIVIEW_DATA_NICKNAME}"
    config["MULTIVIEW_DATA_NAME"] = MULTIVIEW_DATA_NAME

    #------------------------------------------------------------------------
    # SPLIT

    SPLIT_DATA_NAME = (
        MULTIVIEW_DATA_NAME
        + f"__SEED_{SEED}__T_{TRAIN_SIZE}_V_{VAL_SIZE}"
    )

    SPLIT_DATE_TYPE = MULTIVIEW_DATA_TYPE

    config["SPLIT_DATA_NAME"] = SPLIT_DATA_NAME
    config["SPLIT_DATE_TYPE"] = SPLIT_DATE_TYPE

    #------------------------------------------------------------------------
    # AUGMENTATION

    AUG_DATE_NAME = SPLIT_DATA_NAME + "__AUG"
    AUG_DATE_TYPE = SPLIT_DATE_TYPE + "__AUG"

    config["AUG_DATE_NAME"] = AUG_DATE_NAME
    config["AUG_DATE_TYPE"] = AUG_DATE_TYPE

    # return config