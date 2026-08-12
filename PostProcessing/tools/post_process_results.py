"""
File:
    post_process_results.py

Description:
    Analyze the results of training/eval splits for a given set of runs.

Author:
    nubby

Date:
    21 Jul 2026

Version:
    1.0.0
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

def legacy_post_process_results(path_results: str):
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
        df_alt_lstm = df_alt.loc[(df_alt["ModelName"] == "lstm") &
                                         (df_alt["Target"] == target)]
        df_alt_tcn = df_alt.loc[(df_alt["ModelName"] == "tcn") &
                                        (df_alt["Target"] == target)]
        df_alt_trans = df_alt.loc[(df_alt["ModelName"] == "transformer") &
                                        (df_alt["Target"] == target)]
        df_alt_lstm = df_alt.loc[(df_alt["ModelName"] == "lstm") &
                                        (df_alt["Target"] == target)]
        df_alt_tcn = df_alt.loc[(df_alt["ModelName"] == "tcn") &
                                        (df_alt["Target"] == target)]
        df_alt_trans = df_alt.loc[(df_alt["ModelName"] == "transformer") &
                                        (df_alt["Target"] == target)]
    
        # Get useful info.
        try:
            rmse_alt_lstm_min = df_alt_lstm["TestRMSE"].min()
            rmse_alt_lstm_min_seed = df_alt_lstm.loc[
                    df_alt_lstm["TestRMSE"] == rmse_alt_lstm_min, "Seed"].iloc[0]
            rmse_alt_tcn_min = df_alt_tcn["TestRMSE"].min()
            rmse_alt_tcn_min_seed = df_alt_tcn.loc[
                    df_alt_tcn["TestRMSE"] == rmse_alt_tcn_min, "Seed"].iloc[0]
            rmse_alt_trans_min = df_alt_trans["TestRMSE"].min()
            rmse_alt_trans_min_seed = df_alt_trans.loc[
                    df_alt_trans["TestRMSE"] == rmse_alt_trans_min, "Seed"].iloc[0]
            """
            rmse_alt_lstm_min = df_alt_lstm["TestRMSE"].min()
            rmse_alt_lstm_min_seed = df_alt_lstm.loc[
                    df_alt_lstm["TestRMSE"] == rmse_alt_lstm_min, "Seed"].iloc[0]
            rmse_alt_tcn_min = df_alt_tcn["TestRMSE"].min()
            rmse_alt_tcn_min_seed = df_alt_tcn.loc[
                    df_alt_tcn["TestRMSE"] == rmse_alt_tcn_min, "Seed"].iloc[0]
            rmse_alt_trans_min = df_alt_trans["TestRMSE"].min()
            rmse_alt_trans_min_seed = df_alt_trans.loc[
                    df_alt_trans["TestRMSE"] == rmse_alt_trans_min, "Seed"].iloc[0]
            """

            perc_e_alt_lstm_min = df_alt_lstm["TestPercentErrorMean"].min()
            perc_e_alt_lstm_min_seed = df_alt_lstm.loc[
                    df_alt_lstm["TestPercentErrorMean"] == perc_e_alt_lstm_min, "Seed"].iloc[0]
            perc_e_alt_tcn_min = df_alt_tcn["TestPercentErrorMean"].min()
            perc_e_alt_tcn_min_seed = df_alt_tcn.loc[
                    df_alt_tcn["TestPercentErrorMean"] == perc_e_alt_tcn_min, "Seed"].iloc[0]
            perc_e_alt_trans_min = df_alt_trans["TestPercentErrorMean"].min()
            perc_e_alt_trans_min_seed = df_alt_trans.loc[
                    df_alt_trans["TestPercentErrorMean"] == perc_e_alt_trans_min, "Seed"].iloc[0]
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
            acc_alt_lstm_min = df_alt_lstm["TestAccuracy"].min()
            acc_alt_lstm_min_seed = df_alt_lstm.loc[
                    df_alt_lstm["TestAccuracy"] == acc_alt_lstm_min, "Seed"].iloc[0]
            acc_alt_tcn_min = df_alt_tcn["TestAccuracy"].min()
            acc_alt_tcn_min_seed = df_alt_tcn.loc[
                    df_alt_tcn["TestAccuracy"] == acc_alt_tcn_min, "Seed"].iloc[0]
            acc_alt_trans_min = df_alt_trans["TestAccuracy"].min()
            acc_alt_trans_min_seed = df_alt_trans.loc[
                    df_alt_trans["TestAccuracy"] == acc_alt_trans_min, "Seed"].iloc[0]

            # Print it.
            logging.info(
                    f"RESULTS FROM CLASSIC MODE (TARGET={target}):\n"
                    f"\tALL: Number of runs:\t\t{n_alt}\n"
                    "\n"
                    f"\tLSTM:\tNumber of runs:\t\t{len(df_alt_lstm.index)}\n"
                    f"\tLSTM:\tAvg test acc:\t\t{df_alt_lstm['TestAccuracy'].mean()}\n"
                    f"\tLSTM:\tSTD test acc:\t\t{df_alt_lstm['TestAccuracy'].std()}\n"
                    f"\tLSTM:\tAvg test RMSE:\t\t{df_alt_lstm['TestRMSE'].mean()}\n"
                    f"\tLSTM:\tSTD test RMSE:\t\t{df_alt_lstm['TestRMSE'].std()}\n"
                    f"\tLSTM:\tAvg test % err:\t\t{df_alt_lstm['TestPercentErrorMean'].mean()}\n"
                    f"\tLSTM:\tSTD test % err:\t\t{df_alt_lstm['TestPercentErrorMean'].std()}\n"
                    f"\tLSTM:\tBest test acc:\t\t{acc_alt_lstm_min} (seed={acc_alt_lstm_min_seed})\n"
                    f"\tLSTM:\tBest test RMSE:\t\t{rmse_alt_lstm_min} (seed={rmse_alt_lstm_min_seed})\n"
                    f"\tLSTM:\tBest test % err:\t{perc_e_alt_lstm_min} (seed={perc_e_alt_lstm_min_seed})\n"
                    "\n"
                    f"\tTCN:\tNumber of runs:\t\t{len(df_alt_tcn.index)}\n"
                    f"\tTCN:\tAvg test acc:\t\t{df_alt_tcn['TestAccuracy'].mean()}\n"
                    f"\tTCN:\tSTD test acc:\t\t{df_alt_tcn['TestAccuracy'].std()}\n"
                    f"\tTCN:\tAvg test RMSE:\t\t{df_alt_tcn['TestRMSE'].mean()}\n"
                    f"\tTCN:\tSTD test RMSE:\t\t{df_alt_tcn['TestRMSE'].std()}\n"
                    f"\tTCN:\tAvg test % err:\t\t{df_alt_tcn['TestPercentErrorMean'].mean()}\n"
                    f"\tTCN:\tSTD test % err:\t\t{df_alt_tcn['TestPercentErrorMean'].std()}\n"
                    f"\tTCN:\tBest test acc:\t\t{acc_alt_tcn_min} (seed={acc_alt_tcn_min_seed})\n"
                    f"\tTCN:\tBest test RMSE:\t\t{rmse_alt_tcn_min} (seed={rmse_alt_tcn_min_seed})\n"
                    f"\tTCN:\tBest test % err:\t{perc_e_alt_tcn_min} (seed={perc_e_alt_tcn_min_seed})\n"
                    "\n"
                    f"\tTransformer:\tNumber of runs:\t\t{len(df_alt_trans.index)}\n"
                    f"\tTransformer:\tAvg test acc:\t\t{df_alt_trans['TestAccuracy'].mean()}\n"
                    f"\tTransformer:\tSTD test acc:\t\t{df_alt_trans['TestAccuracy'].std()}\n"
                    f"\tTransformer:\tAvg test RMSE:\t\t{df_alt_trans['TestRMSE'].mean()}\n"
                    f"\tTransformer:\tSTD test RMSE:\t\t{df_alt_trans['TestRMSE'].std()}\n"
                    f"\tTransformer:\tAvg test % err:\t\t{df_alt_trans['TestPercentErrorMean'].mean()}\n"
                    f"\tTransformer:\tSTD test % err:\t\t{df_alt_trans['TestPercentErrorMean'].std()}\n"
                    f"\tTransformer:\tBest test acc:\t\t{acc_alt_trans_min} (seed={acc_alt_trans_min_seed})\n"
                    f"\tTransformer:\tBest test RMSE:\t\t{rmse_alt_trans_min} (seed={rmse_alt_trans_min_seed})\n"
                    f"\tTransformer:\tBest test % err:\t{perc_e_alt_trans_min} (seed={perc_e_alt_trans_min_seed})\n"
                )
        except IndexError:
            # IndexError occurs when only one target has thus been run.
            continue

def post_process_results(path_results: str):
    """
    post_process_results(path_results)
    """
    # First load all results into a dict.
    df = _load_csv(path=path_results)

    # Then further filter each by model result.
    n = len(df.index)
    logging.info(f"Found {n} runs:\n")

    # Get stats.
    df_lstm = df.loc[(df["ModelName"] == "lstm")]
    df_tcn = df.loc[(df["ModelName"] == "tcn")]
    df_trans = df.loc[(df["ModelName"] == "transformer")]
    
    # Get useful info.
    try:
        rmse_lstm_min = df_lstm["RMSE (mean)"].min()
        rmse_lstm_min_seed = df_lstm.loc[
                df_lstm["RMSE (mean)"] == rmse_lstm_min, "Seed"].iloc[0]
        rmse_tcn_min = df_tcn["RMSE (mean)"].min()
        rmse_tcn_min_seed = df_tcn.loc[
                df_tcn["RMSE (mean)"] == rmse_tcn_min, "Seed"].iloc[0]
        rmse_trans_min = df_trans["RMSE (mean)"].min()
        rmse_trans_min_seed = df_trans.loc[
                df_trans["RMSE (mean)"] == rmse_trans_min, "Seed"].iloc[0]
        # Print it.
        logging.info(
                f"RESULTS FROM {path_results} (TARGET=SBD):\n"
                f"\tALL: Number of runs:\t\t{n}\n"
                "\n"
                f"\tLSTM:\tNumber of runs:\t\t{len(df_lstm.index)}\n"
                f"\tLSTM:\tAvg test RMSE:\t\t{df_lstm['RMSE (mean)'].mean()}\n"
                f"\tLSTM:\tAvg test RMSE STD:\t\t{df_lstm['RMSE (std)'].mean()}\n"
                f"\tLSTM:\tSTD test RMSE:\t\t{df_lstm['RMSE (mean)'].std()}\n"
                f"\tLSTM:\tBest test RMSE:\t\t{rmse_lstm_min} (seed={rmse_lstm_min_seed})\n"
                "\n"
                f"\tTCN:\tNumber of runs:\t\t{len(df_tcn.index)}\n"
                f"\tTCN:\tAvg test RMSE:\t\t{df_tcn['RMSE (mean)'].mean()}\n"
                f"\tTCN:\tAvg test RMSE STD:\t\t{df_tcn['RMSE (std)'].mean()}\n"
                f"\tTCN:\tSTD test RMSE:\t\t{df_tcn['RMSE (mean)'].std()}\n"
                f"\tTCN:\tBest test RMSE:\t\t{rmse_tcn_min} (seed={rmse_tcn_min_seed})\n"
                "\n"
                f"\tTransformer:\tNumber of runs:\t\t{len(df_trans.index)}\n"
                f"\tTransformer:\tAvg test RMSE:\t\t{df_trans['RMSE (mean)'].mean()}\n"
                f"\tTransformer:\tAvg test RMSE STD:\t\t{df_trans['RMSE (std)'].mean()}\n"
                f"\tTransformer:\tSTD test RMSE:\t\t{df_trans['RMSE (mean)'].std()}\n"
                f"\tTransformer:\tBest test RMSE:\t\t{rmse_trans_min} (seed={rmse_trans_min_seed})\n"
            )
    except IndexError:
        pass


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
