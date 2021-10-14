
import pandas as pd

def add_remaining_useful_life(df : pd.DataFrame) -> pd.DataFrame:
    """
        Calculate the remaining usefull life
        The RUL is the result of the difference between
        the max cycle for the engine and the actual cycle
    """
    dfCopy=df.copy()
    max_df=dfCopy.groupby('engineNumber').agg({'cycleNumber':'max'})
    # Get the total number of cycles for each unit
    max_cycle= max_df['cycleNumber']
    # Merge the max cycle back into the original frame
    dfCopy = dfCopy.merge(max_cycle.to_frame(name='max_cycle'), left_on='cycleNumber', right_index=True)
    # Calculate remaining useful life for each row
    remaining_useful_life = dfCopy["max_cycle"] - dfCopy["cycleNumber"]
    dfCopy["RUL"] = remaining_useful_life
    
    # drop max_cycle as it's no longer needed
    result_frame = dfCopy.drop("max_cycle", axis=1)
    return result_frame