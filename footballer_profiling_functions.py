import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import time
import json
import bs4 as bs
import requests
import unicodedata


def rename_lower_index(df, i):
    """
    Renames the lower index of a multi-index column of a dataframe by appending the higher level name to the lower level
    so that the higher level can be removed, while keeping the column names unique.
    :param df: dataframe to be changed
    :param i: index of the column to be changed
    :return: dataframe with edited lower level name for column i
    """
    old = (df.columns[i])
    temp = list(old)
    temp[1] = temp[0] + temp[1]
    new = tuple(temp)
    df.columns = df.columns.values
    df.columns = pd.MultiIndex.from_tuples(df.rename(columns={old: new}))
    return df


def scrape_season(year):
    """
    Scrapes player data of single season of top 5 leagues from multiple tables from fbref.com and saves preselected
    statistics in a dataframe.
    :param year: start year of season to scrape (2021 for season 2021/22)
    :return: dataframe with combined player statistics for every player
    """
    season = f"{year}-{year + 1}"

    # fbref-sites to be scraped
    shooting = pd.read_html(f"https://fbref.com/en/comps/Big5/{season}/shooting/players/"
                            f"{season}-Big-5-European-Leagues-Stats")
    passing = pd.read_html(f"https://fbref.com/en/comps/Big5/{season}/passing/players/"
                           f"{season}-Big-5-European-Leagues-Stats")
    passing_types = pd.read_html(f"https://fbref.com/en/comps/Big5/{season}/passing_types/players/"
                                 f"{season}-Big-5-European-Leagues-Stats")
    gca = pd.read_html(f"https://fbref.com/en/comps/Big5/{season}/gca/players/"
                       f"{season}-Big-5-European-Leagues-Stats")
    defense = pd.read_html(f"https://fbref.com/en/comps/Big5/{season}/defense/players/"
                           f"{season}-Big-5-European-Leagues-Stats")
    possession = pd.read_html(f"https://fbref.com/en/comps/Big5/{season}/possession/players/"
                              f"{season}-Big-5-European-Leagues-Stats")
    misc = pd.read_html(f"https://fbref.com/en/comps/Big5/{season}/misc/players/"
                        f"{season}-Big-5-European-Leagues-Stats")
    team_poss = pd.read_html(f"https://fbref.com/en/comps/Big5/{season}/possession/squads/"
                             f"{season}-Big-5-European-Leagues-Stats")

    drop_all = ['Rk', 'Nation', 'Born', 'Matches']  # unnecessary columns to be dropped later

    # shooting statistics
    shooting = shooting[0]
    shooting.columns = shooting.columns.droplevel()
    shooting = shooting.drop(drop_all, axis=1)
    shooting = shooting.drop(['Gls', 'SoT', 'Sh/90', 'SoT/90', 'G/Sh', 'G/SoT', 'FK', 'PK', 'PKatt', 'xG', 'npxG',
                              'G-xG'], axis=1)
    shooting = shooting.loc[shooting["Player"] != "Player"]  # to remove rows with column names

    # passing statistics
    passing = passing[0]
    # convert multilevel indexing to single level
    for i in range(len(passing.columns)):
        if not list(passing.columns[i])[0].startswith("Un"):
            passing = rename_lower_index(passing, i)
    passing.columns = passing.columns.droplevel()
    passing = passing.drop(drop_all, axis=1)
    if "Cmp" in passing.columns:  # because of inconsistent top-level of column
        passing.rename(columns={"Cmp": "TotalCmp"}, inplace=True)
    passing = passing.drop(['ShortCmp', 'MediumCmp', 'LongCmp', 'Ast', 'A-xAG'], axis=1)
    passing = passing.loc[passing["Player"] != "Player"]

    # pass types statistics
    passing_types = passing_types[0]
    for i in range(len(passing_types.columns)):
        if not list(passing_types.columns[i])[0].startswith("Un"):
            passing_types = rename_lower_index(passing_types, i)
    passing_types.columns = passing_types.columns.droplevel()
    passing_types = passing_types.drop(drop_all, axis=1)
    drop_types = []
    for c in passing_types.columns:
        if c.startswith("Corner") or c.startswith("Outcomes"):
            drop_types.append(c)
    passing_types = passing_types.drop(drop_types, axis=1)
    passing_types = passing_types.drop(['Att', 'Pass TypesDead', 'Pass TypesFK', 'Pass TypesTB', 'Pass TypesCK',
                                        "Pass TypesTI"], axis=1)
    passing_types = passing_types.loc[passing_types["Player"] != "Player"]

    # shot creating actions statistics
    gca = gca[0]
    for i in range(len(gca.columns)):
        if not list(gca.columns[i])[0].startswith("Un"):
            gca = rename_lower_index(gca, i)
    gca.columns = gca.columns.droplevel()
    gca = gca.drop(drop_all, axis=1)
    drop_types = []
    for c in gca.columns:
        if c.startswith("GCA"):
            drop_types.append(c)
    gca = gca.drop(drop_types, axis=1)
    gca = gca.drop(['SCASCA', 'SCASCA90', 'SCA TypesPassDead', 'SCA TypesSh', 'SCA TypesFld'], axis=1)
    gca = gca.loc[gca["Player"] != "Player"]

    # defensive statistics
    defense = defense[0]
    for i in range(len(defense.columns)):
        if not list(defense.columns[i])[0].startswith("Un"):
            defense = rename_lower_index(defense, i)
    defense.columns = defense.columns.droplevel()
    defense = defense.drop(drop_all, axis=1)
    defense = defense.drop(['Vs DribblesTkl', 'Vs DribblesPast', 'Tkl+Int', 'Err'], axis=1)
    defense = defense.loc[defense["Player"] != "Player"]

    # possession statistics
    possession = possession[0]
    for i in range(len(possession.columns)):
        if not list(possession.columns[i])[0].startswith("Un"):
            possession = rename_lower_index(possession, i)
    possession.columns = possession.columns.droplevel()
    possession = possession.drop(drop_all, axis=1)
    possession = possession.drop(['TouchesLive', 'DribblesSucc', 'DribblesMis'], axis=1)
    possession = possession.loc[possession["Player"] != "Player"]

    # miscellaneous statistics
    misc = misc[0]
    for i in range(len(misc.columns)):
        if list(misc.columns[i])[0].startswith("Aerial"):
            misc = rename_lower_index(misc, i)
    misc.columns = misc.columns.droplevel()
    misc = misc.drop(drop_all, axis=1)
    misc = misc.loc[misc["Player"] != "Player"]
    # calculate total numbers of aerial duels
    misc['Aerial DuelsWon'] = pd.to_numeric(misc['Aerial DuelsWon'])
    misc['Aerial DuelsLost'] = pd.to_numeric(misc['Aerial DuelsLost'])
    misc['Aerial Duels'] = misc['Aerial DuelsWon'] + misc['Aerial DuelsLost']
    misc = misc.drop(['CrdY', 'CrdR', '2CrdY', 'Off', 'Crs', 'Int', 'TklW', 'PKwon', 'PKcon', 'OG', 'Aerial DuelsWon',
                      'Aerial DuelsLost'], axis=1)

    # merging player data to one dataframe
    to_merge = [passing, passing_types, gca, defense, possession, misc]
    common_cols = ["Player", "Pos", "Squad", "Comp", "Age", "90s"]
    df = shooting
    for frame in to_merge:
        df = pd.merge(df, frame, on=common_cols)

    # adding team possession to player data
    team_poss = team_poss[0]
    team_poss.columns = team_poss.columns.droplevel()
    team_poss = team_poss[["Squad", "Comp", "Poss"]]
    team_poss.rename(columns={"Poss": "Team_Poss"}, inplace=True)
    df = df.merge(team_poss, how="left", on=["Squad", "Comp"])

    return df


def get_setup_data(end_season):
    """
    Scrapes data of multiple seasons, starting 2017/18, with 30 second pause between seasons (fbref has 20 requests in
    60 seconds limit) and combines the data into a single dataframe.
    :param end_season: year final season to be scraped ends (2022 for final season 2021/22)
    :return: none - saves dataframe to file basic_data.csv
    """
    print("downloading data")
    number_of_seasons = end_season - 2016
    print(f"0/{number_of_seasons}")
    df, duration = scrape_season(2017)
    df["Season"] = 2017
    print(f"1/{number_of_seasons}")
    for year in range(2018, end_season, 1):
        time.sleep(30)
        new_df = scrape_season(year)
        new_df["Season"] = year  # add season column to identify the season the data belongs to
        df = pd.concat([df, new_df])
        print(f"{year - 2016}/{number_of_seasons}")
    df.to_csv("basic_data.csv", index=False)
    print("download successful")


def get_current_season_data(season):
    """
    Scrapes single season data and saves it into file. To be used for season to be scouted.
    :param season: start year of season to be scraped (2022 for 2022/23)
    :return: none - saves dataframe in file current_season_data.csv
    """
    print("downloading data")
    df = scrape_season(season)
    df.to_csv("current_season_data.csv", index=False)
    print("download successful")


def alter_columns_per_x(df):
    """
    Calculates the relative statistics (per action and per 90 minutes (adjusted)) out of the absolute statistics.
    :param df: dataframe to be altered
    :return: dataframe with relative statistics
    """
    df["Comp"] = df["Comp"].astype(str).str.split(' ', 1).str[1]  # edit competition names
    df = df.fillna(0)
    df = df.replace("nan", 0)
    df["Age"] = df["Age"].astype(str).str.split('-').str[0]
    for column in df:
        if column not in ["Player", "Pos", "Squad", "Comp"]:
            df[column] = pd.to_numeric(df[column])
    # remove goalkeepers, players with too little minutes and incomplete rows
    mask_remove_gk_and_low_minutes = (df.Pos != "GK") & (df["90s"] >= 5) & (df["TouchesTouches"] > 0)
    df = df.loc[mask_remove_gk_and_low_minutes]

    # create additional metrics
    df["Def_Actions"] = df["TacklesTkl"] + df["Vs DribblesAtt"] + df["BlocksBlocks"] + df["Int"] + df["Clr"]
    df["Team_Poss"] = df["Team_Poss"] / 100
    df["Opp_Poss"] = 1 - df["Team_Poss"]

    df.rename(columns={"TouchesTouches": "Touches"}, inplace=True)

    # converting shots to shots per touch
    df["Sh"] = df["Sh"] / df["Touches"]
    df.rename(columns={"Sh": "Sh_pT"}, inplace=True)

    dist_median = df["Dist"].median()  # replace Dist for players with 0 shots (=0 Dist) with median Dist
    df['Dist'].mask(df['Dist'] == 0, dist_median, inplace=True)

    # converting pass statistics to per pass or per completed pass
    df["TotalTotDist"] = df["TotalTotDist"] / df["TotalCmp"]
    df["TotalPrgDist"] = df["TotalPrgDist"] / df["TotalCmp"]
    df["ShortAtt"] = df["ShortAtt"] / df["TotalAtt"]
    df["MediumAtt"] = df["MediumAtt"] / df["TotalAtt"]
    df["LongAtt"] = df["LongAtt"] / df["TotalAtt"]
    df["xAG_p90a"] = df["xAG"] / (df["90s"] * df["Team_Poss"])
    df["xAG"] = df["xAG"] / df["KP"]
    df["xA_p90a"] = df["xA"] / (df["90s"] * df["Team_Poss"])
    df["xA"] = df["xA"] / df["TotalCmp"]
    df["KP_p90a"] = df["KP"] / (df["90s"] * df["Team_Poss"])
    df["KP"] = df["KP"] / df["TotalCmp"]
    df["Pass_Att3rd_p90a"] = df["1/3"] / (df["90s"] * df["Team_Poss"])
    df["1/3"] = df["1/3"] / df["TotalCmp"]
    df["PPA_p90a"] = df["PPA"] / (df["90s"] * df["Team_Poss"])
    df["PPA"] = df["PPA"] / df["TotalCmp"]
    df["CrsPA"] = df["CrsPA"] / df["Pass TypesCrs"]
    df["Prog"] = df["Prog"] / df["TotalCmp"]
    df["Pass TypesSw"] = df["Pass TypesSw"] / df["TotalAtt"]
    df["Pass TypesCrs"] = df["Pass TypesCrs"] / df["TotalAtt"]

    df["TotalAtt"] = df["TotalAtt"] / df["Touches"]

    df = df.drop(["TotalCmp"], axis=1)
    df.rename(
        columns={"TotalAtt": "Pass_Att_pTouch", "TotalCmp%": "Pass_Cmp%", "TotalTotDist": "Pass_Total_Dist_pCPass",
                 "TotalPrgDist": "Pass_Prg_Dist_pCPass", "ShortAtt": "Pass_Short_Att_pPass",
                 "ShortCmp%": "Pass_Short_Cmp%", "MediumAtt": "Pass_Medium_Att_pPass",
                 "MediumCmp%": "Pass_Medium_Cmp%", "LongAtt": "Pass_Long_Att_pPass", "LongCmp%": "Pass_Long_Cmp%",
                 "xAG": "xAG_pKP", "xA": "xA_pCPass", "KP": "KP_pCPass", "1/3": "Pass_Att3rd_pCPass",
                 "PPA": "PPA_pCPass",
                 "CrsPA": "Cmp_Crs_PA_pCrs", "Prog": "Prog_Pass_pPass", "Pass TypesSw": "Switch_pPass",
                 "Pass TypesCrs": "Crs_pPass"}, inplace=True)

    # converting sca statistics to per relevant action
    df["SCA_LivePass_p90a"] = df["SCA TypesPassLive"] / (df["90s"] * df["Team_Poss"])
    df["SCA TypesPassLive"] = df["SCA TypesPassLive"] / df["Pass TypesLive"]
    df["SCA_Drib_p90a"] = df["SCA TypesDrib"] / (df["90s"] * df["Team_Poss"])
    df["SCA TypesDrib"] = df["SCA TypesDrib"] / df["DribblesAtt"]
    df["SCA_Def_p90a"] = df["SCA TypesDef"] / (df["90s"] * df["Team_Poss"])
    df["SCA TypesDef"] = df["SCA TypesDef"] / df["Def_Actions"]

    df = df.drop(["Pass TypesLive"], axis=1)
    df.rename(columns={"SCA TypesPassLive": "SCA_LivePass_pPass", "SCA TypesDrib": "SCA_Drib_pDrib",
                       "SCA TypesDef": "SCA_Def_pDefAct"}, inplace=True)

    # converting defensive statistics to per defensive actions
    df["TacklesTklW"] = df["TacklesTklW"] / df["TacklesTkl"]
    df["TacklesDef 3rd"] = df["TacklesDef 3rd"] / df["TacklesTkl"]
    df["TacklesMid 3rd"] = df["TacklesMid 3rd"] / df["TacklesTkl"]
    df["TacklesAtt 3rd"] = df["TacklesAtt 3rd"] / df["TacklesTkl"]
    df["Vs DribblesAtt"] = df["Vs DribblesAtt"] / df["Def_Actions"]
    df["BlocksSh"] = df["BlocksSh"] / df["BlocksBlocks"]
    df["BlocksPass"] = df["BlocksPass"] / df["BlocksBlocks"]
    df["Blocks_p90a"] = df["BlocksBlocks"] / (df["90s"] * df["Opp_Poss"])
    df["BlocksBlocks"] = df["BlocksBlocks"] / df["Def_Actions"]
    df["Int_p90a"] = df["Int"] / (df["90s"] * df["Opp_Poss"])
    df["Int"] = df["Int"] / df["Def_Actions"]
    df["Clr_p90a"] = df["Clr"] / (df["90s"] * df["Opp_Poss"])
    df["Clr"] = df["Clr"] / df["Def_Actions"]

    df.rename(
        columns={"TacklesTklW": "TklW%", "TacklesDef 3rd": "Tkl_Def_3rd_pTkl", "TacklesMid 3rd": "Tkl_Mid_3rd_pTkl",
                 "TacklesAtt 3rd": "Tkl_Att_3rd_pTkl", "Vs DribblesAtt": "Tkl_vsDrib_pDefAct",
                 "Vs DribblesTkl%": "Tkl_vsDrib%", "BlocksSh": "BlocksSh_pBlock", "BlocksPass": "BlocksPass_pBlock",
                 "BlocksBlocks": "Blocks_pDefAct", "Int": "Int_pDefAct", "Clr": "Clr_pDefAct"}, inplace=True)

    # converting possession statistics to per touch
    df["TouchesDef Pen"] = df["TouchesDef Pen"] / df["Touches"]
    df["TouchesDef 3rd"] = df["TouchesDef 3rd"] / df["Touches"]
    df["TouchesMid 3rd"] = df["TouchesMid 3rd"] / df["Touches"]
    df["TouchesAtt 3rd"] = df["TouchesAtt 3rd"] / df["Touches"]
    df["TouchesAtt Pen"] = df["TouchesAtt Pen"] / df["Touches"]
    df["DribblesAtt"] = df["DribblesAtt"] / df["Touches"]
    df["DribblesDis"] = df["DribblesDis"] / df["Touches"]
    df["ReceivingProg"] = df["ReceivingProg"] / df["ReceivingRec"]
    df["ReceivingRec"] = df["ReceivingRec"] / (df["90s"] * df["Team_Poss"])

    df.rename(columns={"TouchesDef Pen": "Touches_Def_Pen_pTouch", "TouchesDef 3rd": "Touches_Def_3rd_pTouch",
                       "TouchesMid 3rd": "Touches_Mid_3rd_pTouch", "TouchesAtt 3rd": "Touches_Att_3rd_pTouch",
                       "TouchesAtt Pen": "Touches_Att_Pen_pTouch", "DribblesAtt": "Dribbles_Att_pTouch",
                       "DribblesDis": "Dispossessed_pTouch", "ReceivingProg": "Prog_Pass_Rec_pRec",
                       "ReceivingRec": "Pass_Rec_p90a"}, inplace=True)

    # converting misc and aerial statistics to per 90 (adjusted)
    df["Fls_pDefAct"] = df["Fls"] / df["Def_Actions"]
    df["Fls"] = df["Fls"] / (df["90s"] * df["Opp_Poss"])
    df["Fld_pTouch"] = df["Fld"] / df["Touches"]
    df["Fld"] = df["Fld"] / (df["90s"] * df["Team_Poss"])
    df["Recov"] = df["Recov"] / df["90s"]
    df["Aerial Duels"] = df["Aerial Duels"] / df["90s"]

    df["TacklesTkl"] = df["TacklesTkl"] / df["Def_Actions"]

    df.rename(columns={"Fls": "Fls_p90a", "Fld": "Fld_p90a", "Recov": "Recov_p90", "Aerial Duels": "Aerial Duels_p90"},
              inplace=True)

    # convert touches and defensive actions to per 90 (adjusted)
    df["Touches"] = df["Touches"] / (df["90s"] * df["Team_Poss"])
    df["Def_Actions"] = df["Def_Actions"] / (df["90s"] * df["Opp_Poss"])

    df.rename(columns={"Touches": "Touches_p90a", "Def_Actions": "Def_Actions_p90a", "TacklesTkl": "Tackles_pDefAct"},
              inplace=True)

    df = df.drop(["Opp_Poss", "Team_Poss"], axis=1)

    df = df.fillna(0)

    df.reset_index(inplace=True, drop=True)

    return df


def create_min_max(df, filename):
    """
    Creates file with min and max value for each column of dataframe, for later use of values in normalization.
    :param df: dataframe of which max and min values are to be saved
    :param filename: name of file to be saved
    :return: none - saves data in file
    """
    min_max_list = [[], []]
    for column in df.columns:
        min_max_list[0].append(df[column].min())
        min_max_list[1].append(df[column].max())

    min_max_df = pd.DataFrame(min_max_list, columns=df.columns)
    min_max_df.to_csv(filename + ".csv", index=False)


def create_standardized(df, filename):
    """
    Creates file with mean and standard deviation value for each column of dataframe, for later use of values in
    standardization.
    :param df: dataframe of which mean and standard deviation values are to be saved
    :param filename: name of file to be saved
    :return: none - saves data in file
    """
    paramlist = [[], []]
    for column in df.columns:
        paramlist[0].append(df[column].mean())
        paramlist[1].append(np.std(df[column]))

    standardized = pd.DataFrame(paramlist, columns=df.columns)
    standardized.to_csv(filename + ".csv", index=False)


def min_max_normalization(df, column, min_max_df):
    """
    Normalizes single column with min and max value saved in file.
    :param df: dataframe to be normalized
    :param column: name of column to be normalized
    :param min_max_df: dataframe with saved values for normalization
    :return: dataframe with normalized column
    """
    min = min_max_df.loc[0, column]
    max = min_max_df.loc[1, column]
    df[column] = (df[column] - min) / (max - min)
    return df


def normalize_df(df_to_normalize, filename):
    """
    Normalizes whole dataframe with min and max values saved in file.
    :param df_to_normalize: dataframe to be normalized
    :param filename: file containing dataframe with saved values for normalization
    :return: normalized dataframe
    """
    df = df_to_normalize.copy()
    min_max_df = pd.read_csv(filename)
    for column in df.columns:
        df = min_max_normalization(df, column, min_max_df)
    return df


def standardization(df, column, standard_df):
    """
    Standardizes single column with mean and std value saved in file.
    :param df: dataframe to be standardized
    :param column: name of column to be standardized
    :param standard_df: dataframe with values to be used in standardization
    :return: dataframe with standardized column
    """
    mean = standard_df.loc[0, column]
    std = standard_df.loc[1, column]
    df[column] = (df[column] - mean) / std
    return df


def standardize_df(df_to_standard, filename):
    """
    Standardizes whole dataframe with mean and std values saved in file.
    :param df_to_standard: dataframe to be standardized
    :param filename: file containing dataframe with saved values for standardization
    :return: standardized dataframe
    """
    #
    df = df_to_standard.copy()
    standard_df = pd.read_csv(filename)
    for column in df.columns:
        df = standardization(df, column, standard_df)
    return df


def splitting_and_weighting(df, weights):
    """
    Split columns into type and quality statistics and weigh player type columns according to importance.
    :param df: dataframe with statistics to be split and weighted
    :param weights: dataframe with weights
    :return: two dataframes, df with weighted player type statistics, player_quality_df with player quality statistics
    """
    with open("columns_dict.json", 'rb') as fp:
        columns_dict = json.load(fp)

    player_quality_df = df[columns_dict.get('player_quality')]
    player_quality_df.loc[:, "Fls_p90a"] = player_quality_df.loc[:, "Fls_p90a"] * -1

    df = df[columns_dict.get('player_type')]
    df["Fls_pDefAct"] = df["Fls_pDefAct"] * -1

    weights = weights[columns_dict.get('weights')]

    for column in columns_dict.get('shooting'):
        df[column] = df[column] * weights["Sh_pT"]
    for column in columns_dict.get('attacking_pass') + columns_dict.get('passing'):
        df[column] = df[column] * weights["Pass_Att_pTouch"]
    for column in columns_dict.get('defense'):
        df[column] = df[column] * weights["Def_Actions_p90a"]
    df["SCA_Drib_pDrib"] = df["SCA_Drib_pDrib"] * weights["Dribbles_Att_pTouch"]
    for column in columns_dict.get('carrying') + ["Sh_pT", "Pass_Att_pTouch", "Aerial Duels_p90"]:
        df[column] = df[column] * weights["Touches_p90a"]

    return df, player_quality_df


def create_clusters(df, k):
    """
    Creates clusters of player types using k-means and returns centroids as player type prototypes.
    :param df: dataframe with players to cluster
    :param k: number of clusters
    :return: dataframe of cluster centroids representing player types
    """
    km = KMeans(k)
    km.fit(df)
    return pd.DataFrame(km.cluster_centers_, columns=df.columns)


def distance_to_centroids(player_df, centroids):
    """
    Calculates euclidean distance of players to each centroid to determine fitness to player types.
    :param player_df: dataframe of players
    :param centroids: dataframe of player types
    :return: dataframe of euclidean distances of each player to every player type
    """
    result = []
    for index, row in player_df.iterrows():
        to_add = []
        for i in range(len(centroids)):
            to_add.append(np.linalg.norm(row - centroids.iloc[i]))
        result.append(to_add)
    return pd.DataFrame(result, columns=range(len(centroids)))


def scale_rows_to_equal_sum(df, scale_to):
    """
    Scales every entry in dataframe df so that the sum of every row equals scale_to.
    :param df: dataframe with rows to be scaled
    :param scale_to: value each row sum should add up to
    :return: none - saves new values directly into player type
    """
    for i in range(len(df)):
        sum = df.iloc[i].sum()
        df.iloc[i] = df.iloc[i] * (scale_to / sum)


def calc_quality_for_type(centroids, df):
    """
    Calculates quality of every player for every player type.
    :param centroids: dataframe containing player types
    :param df: dataframe containing players and their quality describing statistics.
    :return: dataframe of quality values for every player for each player type
    """
    quality = []
    # filter columns in centroids matching the columns in df
    cent_edit = pd.DataFrame([centroids.loc[:, "Sh_pT"]] * 3 +
                             [centroids.loc[:, "Pass_Att_pTouch"], centroids.loc[:, "Pass_Short_Att_pPass"],
                              centroids.loc[:, "Pass_Medium_Att_pPass"], centroids.loc[:, "Pass_Long_Att_pPass"]] +
                             [centroids.loc[:, "Pass_Att_pTouch"]] * 5 +
                             [centroids.loc[:, "Crs_pPass"], centroids.loc[:, "Pass_Att_pTouch"],
                              centroids.loc[:, "Dribbles_Att_pTouch"], centroids.loc[:, "Tackles_pDefAct"],
                              centroids.loc[:, "Tackles_pDefAct"], centroids.loc[:, "Tkl_vsDrib_pDefAct"],
                              centroids.loc[:, "Blocks_pDefAct"], centroids.loc[:, "Int_pDefAct"],
                              centroids.loc[:, "Clr_pDefAct"], centroids.loc[:, "Dribbles_Att_pTouch"],
                              centroids.loc[:, "Pass_Rec_p90a"], centroids.loc[:, "Fls_pDefAct"],
                              centroids.loc[:, "Fld_pTouch"], centroids.loc[:, "Recov_p90"],
                              centroids.loc[:, "Aerial Duels_p90"]])
    cent_edit = cent_edit.T
    cent_edit.to_csv("edited_centroids.csv", index=False)

    scale_rows_to_equal_sum(cent_edit, 10)  # scale rows to avoid advantages for certain player types
    # for every player
    for index, row in df.iterrows():
        prod = []
        # calculate quality for each player type
        for index_c, centroid in cent_edit.iterrows():
            prod.append(np.dot(row, centroid))
        quality.append(prod)
    return pd.DataFrame(quality)


def calc_quality_for_player(type, df):
    """
    Calculates quality of every player in his playing style.
    :param type: dataframe containing player type statistics for every player
    :param df: dataframe containing players and their quality describing statistics.
    :return: list of quality values for every player
    """
    quality = []
    # filter columns in centroids matching the columns in df
    cent_edit = pd.DataFrame([type.loc[:, "Sh_pT"]] * 3 +
                             [type.loc[:, "Pass_Att_pTouch"], type.loc[:, "Pass_Short_Att_pPass"],
                              type.loc[:, "Pass_Medium_Att_pPass"], type.loc[:, "Pass_Long_Att_pPass"]] +
                             [type.loc[:, "Pass_Att_pTouch"]] * 5 +
                             [type.loc[:, "Crs_pPass"], type.loc[:, "Pass_Att_pTouch"],
                              type.loc[:, "Dribbles_Att_pTouch"], type.loc[:, "Tackles_pDefAct"],
                              type.loc[:, "Tackles_pDefAct"], type.loc[:, "Tkl_vsDrib_pDefAct"],
                              type.loc[:, "Blocks_pDefAct"], type.loc[:, "Int_pDefAct"],
                              type.loc[:, "Clr_pDefAct"], type.loc[:, "Dribbles_Att_pTouch"],
                              type.loc[:, "Pass_Rec_p90a"], type.loc[:, "Fls_pDefAct"],
                              type.loc[:, "Fld_pTouch"], type.loc[:, "Recov_p90"],
                              type.loc[:, "Aerial Duels_p90"]])
    cent_edit = cent_edit.T

    scale_rows_to_equal_sum(cent_edit, 10)  # scale rows to avoid advantages for certain player types
    # for every player
    for index, row in df.iterrows():
        # calculate quality for individual player
        for index_c, centroid in cent_edit.iterrows():
            if index == index_c:
                quality.append(np.dot(row, centroid))
    return quality


def normalize_centroids(centroids):
    """
    Normalizes centroids to convert them to exclusively positive values.
    :param centroids: dataframe containing player types
    :return: dataframe with normalized centroids / player types
    """
#   for column in centroids.iloc[:, 8:].columns:  # player type columns excluding positional columns
    for column in centroids.columns:
        max_cent = centroids[column].max()
        min_cent = centroids[column].min()
        centroids[column] = (centroids[column] - min_cent) / (max_cent - min_cent)
    return centroids


def get_market_values(df):
    """
    Scrapes market values from transfermarkt.de and adds them to the dataframe.
    :param df: dataframe of players
    :return: list with market values
    """
    result = []
    headers = {"User-Agent":"Mozilla/5.0"}
    max_count = len(df)
    count = 0
    for index, row in df.iterrows():
        count += 1
        print(f"Player {count}/{max_count}")
        value = -1  # default value if no matching player was found
        name = row["Player"]
        age = row["Age"]
        search_name = name.replace(" ","+")
        name = unicodedata.normalize('NFKD', name).encode('ASCII', 'ignore')    # changes accents etc to "normal" letters
        url = f"https://www.transfermarkt.de/schnellsuche/ergebnis/schnellsuche?query={search_name}"

        # scraping of transfermarkt.de 's quick search
        player_Site = requests.get(url, headers=headers)
        player_Soup = bs.BeautifulSoup(player_Site.content, 'html.parser')
        player_Site.close()

        tm_people_list = player_Soup.find_all("table",{"class":"items"})    # found entries
        if len(tm_people_list) == 0:
            result.append(-2)   # no entry found
            continue

        tm_player_list =  tm_people_list[0]    # table with found players
        tm_players = tm_player_list.find_all("tr")  # rows of table (index 0 is table header)
        if tm_players[0].text != "\nName / VereinPositionVereinAlterNat.MarktwertBerater":
            result.append(-3)   # no player found
            continue

        for i in range(1, len(tm_players),3):   # 3 entries for every player, first is whole
            tm_row = tm_players[i].find_all("td",{"class":["hauptlink","zentriert"]})
            found_name = tm_row[0].text
            found_name = unicodedata.normalize('NFKD', found_name).encode('ASCII', 'ignore')    # changes accents etc to "normal" letters
            found_age = tm_row[3].text
            found_age = found_age.replace("-","0")
            found_age = found_age.replace("k. A.", "0")
            found_value = tm_row[5].text
            found_value = found_value.replace("-","0.00 Mio. €")
            if (found_name in name or name in found_name) and age == int(found_age):    # looking for players with matching name and age
                found_value = found_value.split()
                if found_value[1] == "Mio.":
                    value = float(found_value[0].replace(",","."))
                elif found_value[1] == "Tsd.":
                    value = round(float(found_value[0].replace(",",""))/1000,3)
                break

        result.append(value)

    return result


def setup_clusters(end_season, k):
    """
    If user confirms to run function, it scrapes data, edits it and creates new player type clusters.
    Otherwise new setup is cancelled.
    :param end_season: year final season to be scraped ends (2022 for season 2021/22)
    :param k: number of clusters / player types to be created
    :return: none - if not cancelled it saves new clusters to file centroids.csv
    """
    # check if user really wants to set up new player types
    while True:
        sure = input("Running this function generates new player type clusters. Those have to be labelled by hand. \n"
                     "Are you sure you want to generate new clusters? (y/n)?")
        if sure == "n":
            print("setup cancelled")
            return
        elif sure == "y":
            break
        else:
            continue
    # scrapes new data and edits it
#    get_setup_data(end_season)
    df = pd.read_csv("basic_data.csv")
    df = alter_columns_per_x(df)

    basic_info = df[["Player", "Pos", "Squad", "Comp", "Age", "Season", "90s"]]  # player identification data
    cols_to_normalize = [col for col in df.columns if col not in basic_info.columns]

    # standardizes, weighs and splits data
    create_min_max(df[cols_to_normalize], "min_max_player_type_columns")
    create_standardized(df[cols_to_normalize], "standardize_player_type_columns")
    weights = normalize_df(df[cols_to_normalize], "min_max_player_type_columns.csv")
    df = standardize_df(df[cols_to_normalize], "standardize_player_type_columns.csv")

    df, player_quality_df = splitting_and_weighting(df, weights)

    # generates clusters
    player_types = create_clusters(df, k)
    player_types.to_csv("centroids.csv", index=False)

    print("setup completed")


def setup_current_season(season,market_values = False):
    """
    Setting up player type fitness and quality for every player of current season data.
    :param market_values: boolean value whether market values from transfermarkt.de should be included
    :return: none - saves fitness and quality table in file scouting.csv
    """
    print("Loading data.")
    # scrapes new data and edits it
    df = get_current_season_data(season)
    df = alter_columns_per_x(df)

    basic_info = df[["Player", "Pos", "Squad", "Comp", "Age", "90s"]]  # player identification data
    cols_to_normalize = [col for col in df.columns if col not in basic_info.columns]

    # standardizes, weighs and splits data
    weights = normalize_df(df[cols_to_normalize], "min_max_player_type_columns.csv")
    df = standardize_df(df[cols_to_normalize], "standardize_player_type_columns.csv")

    df, player_quality_df = splitting_and_weighting(df, weights)
    player_quality_df.to_csv("player_quality_data.csv", index=False)

    player_types = pd.read_csv("centroids.csv")
    print("Calculating Player Fitness.")
    distance = distance_to_centroids(df, player_types)  # calculate player type fitness
    distance = ((1-(distance/distance.max()))*100)  # edit distance to more intuitive scale of 0-100
    distance.to_csv("soft_player_types.csv", index=False)

    player_types = normalize_centroids(player_types)
    print("Calculating Player Quality")
    quality = calc_quality_for_type(player_types, player_quality_df)  # calculate player type quality

    # calculates player quality for his playing style
    df = normalize_centroids(df)
    individual_quality = calc_quality_for_player(df, player_quality_df)

    # edits quality to scale 0-100
    max_quality = max(individual_quality)
    min_quality = min(individual_quality)
    individual_quality = ((individual_quality-min_quality)/(max_quality-min_quality))*100
    basic_info["Quality"] = individual_quality

    #for column in quality.columns:
    #    quality[column] = ((quality[column]-min_quality)/(max_quality-min_quality))*100
    quality = ((quality-min_quality)/(max_quality-min_quality))*100

    with open("player_type_names.json", 'rb') as fp:
        type_names = json.load(fp)
    type_names = list(type_names.keys())

    # combines player fitness and quality for every player type into single table
    for i in range(len(player_types)):
        name = type_names[i]
        basic_info[f"Fit_{name}"] = distance.iloc[:, i]
        basic_info[f"Qual_{name}"] = quality.iloc[:, i]

    print("Getting Market Values.")
    if market_values:   # inserts market value for every player if wanted, else sets value to 1.0
        basic_info["Value"] = get_market_values(basic_info)
    else:
        basic_info["Value"] = 1.0

    basic_info.to_csv("scouting.csv", index=False)

    print("Setup Completed.")


def scout_player_type(player_type, age_min=0, age_max=99, value_max=10000.0, count=5):
    """
    Scouts players for fitness of specified player type .
    :param player_type: player type to be scouted
    :param age_min: lower age limit (default: 0)
    :param age_max: upper age limit (default: 99)
    :param value_max: upper value limit in million euros (default: 10000)
    :param count: number of players to be shown (default: 5)
    :return: top players for chosen player type and set filters
    """
    data = pd.read_csv("scouting.csv")
    data = data.loc[(data[f"Fit_{player_type}"] < 3) & (data["Age"] >= age_min) & (data["Age"] <= age_max)
                    & (data["Value"] <= value_max)]
    if len(data) < 1:
        print("no players found")
        return
    data = data[["Player", "Pos", "Squad", "Age", f"Fit_{player_type}", f"Qual_{player_type}", "Value"]]
    data.rename(columns={f"Fit_{player_type}": "Fitness", f"Qual_{player_type}": "Quality"}, inplace=True)
    return data.sort_values("Quality", ascending=False).head(count)
