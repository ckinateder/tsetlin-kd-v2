TEACHER_BASELINE_MODEL_PATH = "teacher_baseline.pkl"
STUDENT_BASELINE_MODEL_PATH = "student_baseline.pkl"
TEACHER_CHECKPOINT_PATH = "teacher_checkpoint.pkl"
DISTILLED_MODEL_PATH = "distilled.pkl"
OUTPUT_JSON_PATH = "output.json"
RESULTS_CSV_PATH = "results.csv"
TEST_ACCURACY_PNG_PATH = "test_accuracy.png"
TRAIN_ACCURACY_PNG_PATH = "train_accuracy.png"
TEST_TIME_PNG_PATH = "test_time.png"
TRAIN_TIME_PNG_PATH = "train_time.png"
ACTIVATION_MAPS_PNG_PATH = "activation_maps.png"
DEFAULT_FOLDERPATH = "experiments"
DATASET_FOLDERPATH = "data"
ACC_TEST_TEACHER = "acc_test_teacher"
ACC_TEST_STUDENT = "acc_test_student"
ACC_TEST_DISTILLED = "acc_test_distilled"
ACC_TRAIN_TEACHER = "acc_train_teacher"
ACC_TRAIN_STUDENT = "acc_train_student"
ACC_TRAIN_DISTILLED = "acc_train_distilled"
TIME_TRAIN_TEACHER = "time_train_teacher"
TIME_TRAIN_STUDENT = "time_train_student"
TIME_TRAIN_DISTILLED = "time_train_distilled"
TIME_TEST_TEACHER = "time_test_teacher"
TIME_TEST_STUDENT = "time_test_student"
TIME_TEST_DISTILLED = "time_test_distilled"

PLOT_FIGSIZE = (8, 6)
PLOT_DPI = 300

RESULTS_COLUMNS = [ACC_TEST_TEACHER, ACC_TEST_STUDENT, ACC_TEST_DISTILLED, ACC_TRAIN_TEACHER, ACC_TRAIN_STUDENT,
                           ACC_TRAIN_DISTILLED, TIME_TRAIN_TEACHER, TIME_TRAIN_STUDENT, TIME_TRAIN_DISTILLED,
                           TIME_TEST_TEACHER, TIME_TEST_STUDENT, TIME_TEST_DISTILLED]