TEACHER_BASELINE_MODEL_PATH = "teacher_baseline.pkl"
BASELINE_MODEL_PATH = "baseline.pkl"
TEACHER_CHECKPOINT_PATH = "teacher_checkpoint.pkl"
STUDENT_MODEL_PATH = "student.pkl"
STUDENT_DS_MODEL_PATH = "student_ds.pkl"

OUTPUT_JSON_PATH = "output.json"
RESULTS_CSV_PATH = "results.csv"
AGGREGATED_OUTPUT_JSON_PATH = "aggregated_output.json"
AGGREGATED_RESULTS_CSV_PATH = "aggregated_results.csv"

TEST_ACCURACY_PNG_PATH = "test_accuracy.png"
TRAIN_ACCURACY_PNG_PATH = "train_accuracy.png"
TEST_TIME_PNG_PATH = "test_time.png"
TRAIN_TIME_PNG_PATH = "train_time.png"
ACTIVATION_MAPS_PNG_PATH = "activation_maps.png"
DEFAULT_FOLDERPATH = "experiments"
DATASET_FOLDERPATH = "data"

ACC_TEST_TEACHER = "acc_test_teacher"
ACC_TEST_BASELINE = "acc_test_baseline"
ACC_TEST_STUDENT = "acc_test_student"
ACC_TEST_STUDENT_DS = "acc_test_student_ds"

ACC_TRAIN_TEACHER = "acc_train_teacher"
ACC_TRAIN_BASELINE = "acc_train_baseline"
ACC_TRAIN_STUDENT = "acc_train_student"
ACC_TRAIN_STUDENT_DS = "acc_train_student_ds"

TIME_TRAIN_TEACHER = "time_train_teacher"
TIME_TRAIN_BASELINE = "time_train_baseline"
TIME_TRAIN_STUDENT = "time_train_student"
TIME_TRAIN_STUDENT_DS = "time_train_student_ds"

TIME_TEST_TEACHER = "time_test_teacher"
TIME_TEST_BASELINE = "time_test_baseline"
TIME_TEST_STUDENT = "time_test_student"
TIME_TEST_STUDENT_DS = "time_test_student_ds"

PLOT_FIGSIZE = (8, 5.8)
PLOT_DPI = 300

DISTRIBUTION_RESULTS_COLUMNS = [ACC_TEST_TEACHER, ACC_TEST_BASELINE, ACC_TEST_STUDENT,
                               ACC_TRAIN_TEACHER, ACC_TRAIN_BASELINE, ACC_TRAIN_STUDENT,
                               TIME_TRAIN_TEACHER, TIME_TRAIN_BASELINE, TIME_TRAIN_STUDENT,
                               TIME_TEST_TEACHER, TIME_TEST_BASELINE, TIME_TEST_STUDENT]

CLAUSE_RESULTS_COLUMNS = [ACC_TEST_TEACHER, ACC_TEST_BASELINE, ACC_TEST_STUDENT, ACC_TEST_STUDENT_DS,
                         ACC_TRAIN_TEACHER, ACC_TRAIN_BASELINE, ACC_TRAIN_STUDENT, ACC_TRAIN_STUDENT_DS,
                         TIME_TRAIN_TEACHER, TIME_TRAIN_BASELINE, TIME_TRAIN_STUDENT, TIME_TRAIN_STUDENT_DS,
                         TIME_TEST_TEACHER, TIME_TEST_BASELINE, TIME_TEST_STUDENT, TIME_TEST_STUDENT_DS]
