import pandas as pd
import sys
from sklearn.model_selection import train_test_split

def filter_df(df):
    return df
    '''
    Function to filter audio files based on df columns
    df column options: [age,age_of_english_onset,age_sex,birth_place,english_learning_method,
    english_residence,length_of_english_residence,native_language,other_languages,sex]
    :param df (DataFrame): Full unfiltered DataFrame
    :return (DataFrame): Filtered DataFrame
    '''
"""
    arabic = df[df.native_language_rev.str.contains('arabic')]
    mandarin = df[df.native_language_rev.str.contains('mandarin')]
    english = df[df.native_language_rev.str.contains('english')][:125]

    mandarin = mandarin[mandarin.length_of_english_residence < 10][:125]
    arabic = arabic[arabic.length_of_english_residence < 10][:125]

    print(english.shape)
    print(mandarin.shape)
    print(arabic.shape)
"""
    # df = df.append(english)
    # df = df.append(arabic)
    # df = df.append(mandarin)

    # print(df_new.info(verbose=True))

    # df_new['birth_place'].apply(lambda col: col['birth_place'] = col['birth_place'].str[-1])

def split_people(df,test_size=0.2):
    '''
    Create train test split of DataFrame
    :param df (DataFrame): Pandas DataFrame of audio files to be split
    :param test_size (float): Percentage of total files to be split into test
    :return X_train, X_test, y_train, y_test (tuple): Xs are list of df['language_num'] and Ys are df['native_language']
    '''


    return train_test_split(df['language_num'],df['english_residence'],test_size=test_size) # random_state = 1234


if __name__ == '__main__':
    '''
    Console command example:
    python bio_data.csv
    '''

    csv_file = sys.argv[1]
    df = pd.read_csv(csv_file)
    filtered_df = filter_df(df)
    print(split_people(filtered_df))
