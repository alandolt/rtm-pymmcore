import pandas as pd


class Tracker:
    """Base class for tracking algorithms. Subclasses must implement track_cells() method."""

    def track_cells(self) -> pd.DataFrame:
        raise NotImplementedError("Subclass must implement track_cells() method.")


class TrackerNoTracking(Tracker):
    """Dummy tracker that just joins the old and new dataframe, and doesn't track anything."""

    def track_cells(self, df_old, df_new, metadata):
        if df_old is None:
            return None
        df_tracked = pd.concat([df_old, df_new])
        df_tracked = df_tracked.reset_index(drop=True)
        print(df_tracked.head())
        return df_tracked
