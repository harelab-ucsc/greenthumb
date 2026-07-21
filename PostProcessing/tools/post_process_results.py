"""
File:
    post_process_results.py

Description:
    Analyze the results of training/eval splits for a given set of runs.

Author:
    nubby

Date:
    20 Jul 2026

Version:
    0.0.2
"""
import argparse
import csv
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger("greenthumb")


def _load_csv(path: str) -> pd.DataFrame:
    """
    _load_csv(path) -> data
    """
    data = pd.read_csv(path)
    return data 

def split_alt_and_classic_runs(
        data: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    split_alt_and_classic_runs(data) -> alt, classic
    """
    # Filter out duplicates as well.
    data_alt = data.loc[data["UseClassicMode"] == False].drop_duplicates(
            subset=["ModelName", "Seed", "TestAccuracy"])
    data_classic = data.loc[data["UseClassicMode"] == True].drop_duplicates(
            subset=["ModelName", "Seed", "TestAccuracy"])

    return data_alt, data_classic

def post_process_results(path_results: str):
    """
    post_process_results(path_results)
    """
    # First load all results into a dict.
    df_all = _load_csv(path=path_results)

    # Separate "classic mode" and "alternate" runs.
    df_alt, df_classic = split_alt_and_classic_runs(data=df_all)

    # Then further filter each by model result.
    n_classic = len(df_classic.index)
    n_alt = len(df_alt.index)
    logging.info(f"Found {n_classic} classic, {n_alt} alternative runs:\n")

    for target in ["spr", "sbd"]:
        # Get stats.
        df_classic_lstm = df_classic.loc[(df_classic["ModelName"] == "lstm") &
                                         (df_classic["Target"] == target)]
        df_classic_tcn = df_classic.loc[(df_classic["ModelName"] == "tcn") &
                                        (df_classic["Target"] == target)]
        df_classic_trans = df_classic.loc[(df_classic["ModelName"] == "transformer") &
                                        (df_classic["Target"] == target)]
        df_alt_lstm = df_alt.loc[(df_alt["ModelName"] == "lstm") &
                                        (df_alt["Target"] == target)]
        df_alt_tcn = df_alt.loc[(df_alt["ModelName"] == "tcn") &
                                        (df_alt["Target"] == target)]
        df_alt_trans = df_alt.loc[(df_alt["ModelName"] == "transformer") &
                                        (df_alt["Target"] == target)]
    
        # Get useful info.
        try:
            rmse_classic_lstm_max = df_classic_lstm["TestRMSE"].max()
            rmse_classic_lstm_max_seed = df_classic_lstm.loc[
                    df_classic_lstm["TestRMSE"] == rmse_classic_lstm_max, "Seed"].iloc[0]
            rmse_classic_tcn_max = df_classic_tcn["TestRMSE"].max()
            rmse_classic_tcn_max_seed = df_classic_tcn.loc[
                    df_classic_tcn["TestRMSE"] == rmse_classic_tcn_max, "Seed"].iloc[0]
            rmse_classic_trans_max = df_classic_trans["TestRMSE"].max()
            rmse_classic_trans_max_seed = df_classic_trans.loc[
                    df_classic_trans["TestRMSE"] == rmse_classic_trans_max, "Seed"].iloc[0]
            """
            rmse_alt_lstm_max = df_alt_lstm["TestRMSE"].max()
            rmse_alt_lstm_max_seed = df_alt_lstm.loc[
                    df_alt_lstm["TestRMSE"] == rmse_alt_lstm_max, "Seed"].iloc[0]
            rmse_alt_tcn_max = df_alt_tcn["TestRMSE"].max()
            rmse_alt_tcn_max_seed = df_alt_tcn.loc[
                    df_alt_tcn["TestRMSE"] == rmse_alt_tcn_max, "Seed"].iloc[0]
            rmse_alt_trans_max = df_alt_trans["TestRMSE"].max()
            rmse_alt_trans_max_seed = df_alt_trans.loc[
                    df_alt_trans["TestRMSE"] == rmse_alt_trans_max, "Seed"].iloc[0]
            """

            perc_e_classic_lstm_min = df_classic_lstm["TestPercentErrorMean"].min()
            perc_e_classic_lstm_min_seed = df_classic_lstm.loc[
                    df_classic_lstm["TestPercentErrorMean"] == perc_e_classic_lstm_min, "Seed"].iloc[0]
            perc_e_classic_tcn_min = df_classic_tcn["TestPercentErrorMean"].min()
            perc_e_classic_tcn_min_seed = df_classic_tcn.loc[
                    df_classic_tcn["TestPercentErrorMean"] == perc_e_classic_tcn_min, "Seed"].iloc[0]
            perc_e_classic_trans_min = df_classic_trans["TestPercentErrorMean"].min()
            perc_e_classic_trans_min_seed = df_classic_trans.loc[
                    df_classic_trans["TestPercentErrorMean"] == perc_e_classic_trans_min, "Seed"].iloc[0]
            """
            perc_e_alt_lstm_min = df_alt_lstm["TestPercentErrorMean"].min()
            perc_e_alt_lstm_min_seed = df_alt_lstm.loc[
                    df_alt_lstm["TestPercentErrorMean"] == perc_e_alt_lstm_min, "Seed"]
            perc_e_alt_tcn_min = df_alt_tcn["TestPercentErrorMean"].min()
            perc_e_alt_tcn_min_seed = df_alt_tcn.loc[
                    df_alt_tcn["TestPercentErrorMean"] == perc_e_alt_tcn_min, "Seed"]
            perc_e_alt_trans_min = df_alt_trans["TestPercentErrorMean"].min()
            perc_e_alt_trans_min_seed = df_alt_trans.loc[
                    df_alt_trans["TestPercentErrorMean"] == perc_e_alt_trans_min, "Seed"]
            """

            # Print it.
            logging.info(
                    f"RESULTS FROM CLASSIC MODE (TARGET={target}):\n"
                    f"\tALL: Number of runs:\t\t{n_classic}\n"
                    "\n"
                    f"\tLSTM:\tNumber of runs:\t\t{len(df_classic_lstm.index)}\n"
                    f"\tLSTM:\tAvg test acc:\t\t{df_classic_lstm['TestAccuracy'].mean()}\n"
                    f"\tLSTM:\tSTD test acc:\t\t{df_classic_lstm['TestAccuracy'].std()}\n"
                    f"\tLSTM:\tAvg test RMSE:\t\t{df_classic_lstm['TestRMSE'].mean()}\n"
                    f"\tLSTM:\tSTD test RMSE:\t\t{df_classic_lstm['TestRMSE'].std()}\n"
                    f"\tLSTM:\tAvg test % err:\t\t{df_classic_lstm['TestPercentErrorMean'].mean()}\n"
                    f"\tLSTM:\tSTD test % err:\t\t{df_classic_lstm['TestPercentErrorMean'].std()}\n"
                    f"\tLSTM:\tBest test RMSE:\t\t{rmse_classic_lstm_max} (seed={rmse_classic_lstm_max_seed})\n"
                    f"\tLSTM:\tBest test % err:\t{perc_e_classic_lstm_min} (seed={perc_e_classic_lstm_min_seed})\n"
                    "\n"
                    f"\tTCN:\tNumber of runs:\t\t{len(df_classic_tcn.index)}\n"
                    f"\tTCN:\tAvg test acc:\t\t{df_classic_tcn['TestAccuracy'].mean()}\n"
                    f"\tTCN:\tSTD test acc:\t\t{df_classic_tcn['TestAccuracy'].std()}\n"
                    f"\tTCN:\tAvg test RMSE:\t\t{df_classic_tcn['TestRMSE'].mean()}\n"
                    f"\tTCN:\tSTD test RMSE:\t\t{df_classic_tcn['TestRMSE'].std()}\n"
                    f"\tTCN:\tAvg test % err:\t\t{df_classic_tcn['TestPercentErrorMean'].mean()}\n"
                    f"\tTCN:\tSTD test % err:\t\t{df_classic_tcn['TestPercentErrorMean'].std()}\n"
                    f"\tTCN:\tBest test RMSE:\t\t{rmse_classic_tcn_max} (seed={rmse_classic_tcn_max_seed})\n"
                    f"\tTCN:\tBest test % err:\t{perc_e_classic_tcn_min} (seed={perc_e_classic_tcn_min_seed})\n"
                    "\n"
                    f"\tTransformer:\tNumber of runs:\t\t{len(df_classic_trans.index)}\n"
                    f"\tTransformer:\tAvg test acc:\t\t{df_classic_trans['TestAccuracy'].mean()}\n"
                    f"\tTransformer:\tSTD test acc:\t\t{df_classic_trans['TestAccuracy'].std()}\n"
                    f"\tTransformer:\tAvg test RMSE:\t\t{df_classic_trans['TestRMSE'].mean()}\n"
                    f"\tTransformer:\tSTD test RMSE:\t\t{df_classic_trans['TestRMSE'].std()}\n"
                    f"\tTransformer:\tAvg test % err:\t\t{df_classic_trans['TestPercentErrorMean'].mean()}\n"
                    f"\tTransformer:\tSTD test % err:\t\t{df_classic_trans['TestPercentErrorMean'].std()}\n"
                    f"\tTransformer:\tBest test RMSE:\t\t{rmse_classic_trans_max} (seed={rmse_classic_trans_max_seed})\n"
                    f"\tTransformer:\tBest test % err:\t{perc_e_classic_trans_min} (seed={perc_e_classic_trans_min_seed})\n"
                )
        except IndexError:
            # IndexError occurs when only one target has thus been run.
            continue

if __name__ == "__main__":
    # Logger setup.
    logging.basicConfig(level=logging.INFO)

    # Arg parsing.
    parser = argparse.ArgumentParser(
            description="Digest run results."
        )

    parser.add_argument(
            "--path-results",
            type=str,
            default="artifacts/results.csv",
            help="Path to the file containing run results."
        )
    args = parser.parse_args()

    post_process_results(path_results=args.path_results)
