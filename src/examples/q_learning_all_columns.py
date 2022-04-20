
from src.examples.helpers.load_full_mock_dataset import load_discrete_env, get_ethinicity_hierarchy, get_gender_hierarchy
from src.datasets import ColumnType
from src.spaces.env_type import DiscreteEnvType
from src.spaces.action_space import ActionSpace
from src.spaces.actions import ActionIdentity, ActionStringGeneralize, ActionNumericBinGeneralize

N_BINS = 20
PUNISH_FACTOR = 2.0
MAX_DISTORTION = 0.7
MIN_DISTORTION = 0.4

if __name__ == '__main__':

    column_types = {"NHSno": ColumnType.IDENTIFYING_ATTRIBUTE,
                    "given_name": ColumnType.IDENTIFYING_ATTRIBUTE,
                    "surname": ColumnType.IDENTIFYING_ATTRIBUTE,
                    "gender": ColumnType.QUASI_IDENTIFYING_ATTRIBUTE,
                    "dob": ColumnType.SENSITIVE_ATTRIBUTE,
                    "ethnicity": ColumnType.QUASI_IDENTIFYING_ATTRIBUTE,
                    "education": ColumnType.SENSITIVE_ATTRIBUTE,
                    "salary": ColumnType.SENSITIVE_ATTRIBUTE,
                    "mutation_status": ColumnType.SENSITIVE_ATTRIBUTE,
                    "preventative_treatment": ColumnType.SENSITIVE_ATTRIBUTE,
                    "diagnosis": ColumnType.INSENSITIVE_ATTRIBUTE}

    action_space = ActionSpace(n=8)
    # all the columns that are SENSITIVE_ATTRIBUTE will be kept as they are
    # because currently we have no model
    # also INSENSITIVE_ATTRIBUTE will be kept as is
    action_space.add_many(ActionIdentity(column_name="dob"),
                          ActionIdentity(column_name="education"),
                          ActionIdentity(column_name="salary"),
                          ActionIdentity(column_name="diagnosis"),
                          ActionIdentity(column_name="mutation_status"),
                          ActionIdentity(column_name="preventative_treatment"),
                          ActionStringGeneralize(column_name="ethnicity",
                                                 generalization_table=get_ethinicity_hierarchy()),
                          ActionStringGeneralize(column_name="gender",
                                                 generalization_table=get_gender_hierarchy()))

    env = load_discrete_env(env_type=DiscreteEnvType.TOTAL_DISTORTION_STATE,
                            n_states=N_BINS,
                            min_distortion=MIN_DISTORTION, max_distortion=MAX_DISTORTION,
                            total_min_distortion=MIN_DISTORTION, total_max_distortion=MAX_DISTORTION,
                            punish_factor=PUNISH_FACTOR, column_types=column_types,
                            action_space=action_space)