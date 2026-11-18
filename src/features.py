import pandas as pd


ACTIVITIES = [
    "open_pr",
    "open_issue",
    "issue_comment",
    "pr_review",
    "pr_review_comment",
    "push"
]


def convert_to_utc(df, columns):
    df = df.copy()

    for column in columns:
        if column in df.columns:
            df[column] = pd.to_datetime(
                df[column],
                utc=True,
                errors="coerce"
            )

    return df


def build_activity_events(
    prs,
    issues,
    issue_comments,
    pr_reviews,
    pr_review_comments,
    pushes
):
    prs = convert_to_utc(
        prs,
        ["created_at", "first_seen_at", "merged_at", "closed_at"]
    )

    issues = convert_to_utc(
        issues,
        ["created_at", "first_seen_at", "closed_at"]
    )

    issue_comments = convert_to_utc(
        issue_comments,
        ["created_at"]
    )

    pr_reviews = convert_to_utc(
        pr_reviews,
        ["submitted_at", "first_seen_at", "dismissed_at"]
    )

    pr_review_comments = convert_to_utc(
        pr_review_comments,
        ["created_at"]
    )

    pushes = convert_to_utc(
        pushes,
        ["created_at"]
    )

    # Open PR
    open_pr_events = prs[
        prs["author_id"].notna()
        & prs["created_at"].notna()
    ].copy()

    open_pr_events = open_pr_events.rename(
        columns={
            "author_id": "user_id",
            "created_at": "event_time"
        }
    )

    open_pr_events["activity"] = "open_pr"

    open_pr_events = open_pr_events[
        ["user_id", "repo_id", "event_time",
         "first_seen_at", "activity"]
    ]

    # Open issue
    open_issue_events = issues[
        issues["author_id"].notna()
        & issues["created_at"].notna()
        & ~issues["is_pull_request"].fillna(False)
    ].copy()

    open_issue_events = open_issue_events.rename(
        columns={
            "author_id": "user_id",
            "created_at": "event_time"
        }
    )

    open_issue_events["activity"] = "open_issue"

    open_issue_events = open_issue_events[
        ["user_id", "repo_id", "event_time",
         "first_seen_at", "activity"]
    ]

    # Issue comments
    issue_comment_events = issue_comments[
        issue_comments["author_id"].notna()
        & issue_comments["created_at"].notna()
    ].copy()

    issue_comment_events = issue_comment_events.rename(
        columns={
            "author_id": "user_id",
            "created_at": "event_time"
        }
    )

    issue_comment_events["first_seen_at"] = (
        issue_comment_events["event_time"]
    )
    issue_comment_events["activity"] = "issue_comment"

    issue_comment_events = issue_comment_events[
        ["user_id", "repo_id", "event_time",
         "first_seen_at", "activity"]
    ]

    # PR reviews
    pr_review_events = pr_reviews[
        pr_reviews["reviewer_id"].notna()
        & pr_reviews["submitted_at"].notna()
    ].copy()

    pr_review_events = pr_review_events.rename(
        columns={
            "reviewer_id": "user_id",
            "submitted_at": "event_time"
        }
    )

    pr_review_events["activity"] = "pr_review"

    pr_review_events = pr_review_events[
        ["user_id", "repo_id", "event_time",
         "first_seen_at", "activity"]
    ]

    # PR review comments
    pr_review_comment_events = pr_review_comments[
        pr_review_comments["author_id"].notna()
        & pr_review_comments["created_at"].notna()
    ].copy()

    pr_review_comment_events = pr_review_comment_events.rename(
        columns={
            "author_id": "user_id",
            "created_at": "event_time"
        }
    )

    pr_review_comment_events["first_seen_at"] = (
        pr_review_comment_events["event_time"]
    )
    pr_review_comment_events["activity"] = "pr_review_comment"

    pr_review_comment_events = pr_review_comment_events[
        ["user_id", "repo_id", "event_time",
         "first_seen_at", "activity"]
    ]

    # Pushes
    push_events = pushes[
        pushes["user_id"].notna()
        & pushes["created_at"].notna()
    ].copy()

    push_events = push_events.rename(
        columns={"created_at": "event_time"}
    )

    push_events["first_seen_at"] = push_events["event_time"]
    push_events["activity"] = "push"

    push_events = push_events[
        ["user_id", "repo_id", "event_time",
         "first_seen_at", "activity"]
    ]

    activity_events = pd.concat(
        [
            open_pr_events,
            open_issue_events,
            issue_comment_events,
            pr_review_events,
            pr_review_comment_events,
            push_events
        ],
        ignore_index=True
    )

    activity_events["user_id"] = (
        activity_events["user_id"].astype("int64")
    )

    activity_events = activity_events.sort_values(
        "event_time"
    ).reset_index(drop=True)

    return activity_events


def build_features(users, activity_events, prediction_time, windows=(7, 30, 90)):
    prediction_time = pd.Timestamp(prediction_time)

    if prediction_time.tzinfo is None:
        prediction_time = prediction_time.tz_localize("UTC")
    else:
        prediction_time = prediction_time.tz_convert("UTC")

    # The task concerns fixed cohort users
    features_df = users[
        users["is_cohort"].fillna(False)
    ][
        ["user_id", "github_user_id", "login"]
    ].copy()

    # Only use activity before prediction time
    history = activity_events[
        (activity_events["event_time"] < prediction_time)
        & (activity_events["first_seen_at"] < prediction_time)
    ].copy()

    for window in windows:
        window_start = (prediction_time - pd.Timedelta(days=window))

        window_events = history[
            history["event_time"] >= window_start
        ].copy()

        # Activity count features
        counts = (
            window_events
            .groupby(["user_id", "activity"])
            .size()
            .unstack(fill_value=0)
        )

        counts = counts.reindex(
            columns=ACTIVITIES,
            fill_value=0
        )

        counts.columns = [
            f"{activity}_{window}d"
            for activity in counts.columns
        ]

        features_df = features_df.merge(
            counts,
            left_on="user_id",
            right_index=True,
            how="left"
        )

        # Number of distinct repositories
        distinct_repos = (
            window_events
            .groupby("user_id")["repo_id"]
            .nunique()
            .rename(f"distinct_repos_{window}d")
        )

        features_df = features_df.merge(
            distinct_repos,
            left_on="user_id",
            right_index=True,
            how="left"
        )

        # Number of active days
        window_events["activity_date"] = (
            window_events["event_time"].dt.floor("D")
        )

        active_days = (
            window_events
            .groupby("user_id")["activity_date"]
            .nunique()
            .rename(f"active_days_{window}d")
        )

        features_df = features_df.merge(
            active_days,
            left_on="user_id",
            right_index=True,
            how="left"
        )

    # Recency for each activity type
    for activity in ACTIVITIES:
        activity_history = history[
            history["activity"] == activity
        ]

        last_activity = (
            activity_history
            .groupby("user_id")["event_time"]
            .max()
        )

        days_since = (
            (prediction_time - last_activity)
            .dt.total_seconds()
            / 86_400
        ).rename(f"days_since_last_{activity}")

        features_df = features_df.merge(
            days_since,
            left_on="user_id",
            right_index=True,
            how="left"
        )

    # Overall activity features
    total_events = (
        history
        .groupby("user_id")
        .size()
        .rename("total_historical_events")
    )

    total_repos = (
        history
        .groupby("user_id")["repo_id"]
        .nunique()
        .rename("total_historical_repos")
    )

    features_df = features_df.merge(
        total_events,
        left_on="user_id",
        right_index=True,
        how="left"
    )

    features_df = features_df.merge(
        total_repos,
        left_on="user_id",
        right_index=True,
        how="left"
    )

    # Count features become 0 when user has no activity
    count_columns = [
        column
        for column in features_df.columns
        if (
            column.endswith(("7d", "30d", "90d"))
            or column.startswith("total_")
        )
        and not column.startswith("days_since_")
    ]

    features_df[count_columns] = (
        features_df[count_columns]
        .fillna(0)
        .astype("int64")
    )

    # Large value means no previous activity of this type
    recency_columns = [
        column
        for column in features_df.columns
        if column.startswith("days_since_last_")
    ]

    features_df[recency_columns] = (
        features_df[recency_columns]
        .fillna(9999)
    )

    features_df["prediction_time"] = prediction_time

    return features_df    

def build_labels(users, activity_events, prediction_time, horizon_days=7):
    prediction_time = pd.Timestamp(prediction_time)

    if prediction_time.tzinfo is None:
        prediction_time = prediction_time.tz_localize("UTC")
    else:
        prediction_time = prediction_time.tz_convert("UTC")

    label_end = (
        prediction_time
        + pd.Timedelta(days=horizon_days)
    )

    labels_df = users[
        users["is_cohort"].fillna(False)
    ][
        ["user_id"]
    ].copy()

    future = activity_events[
        (activity_events["event_time"] >= prediction_time)
        & (activity_events["event_time"] < label_end)
    ].copy()

    labels = (
        future
        .assign(label=1)
        .pivot_table(
            index="user_id",
            columns="activity",
            values="label",
            aggfunc="max",
            fill_value=0
        )
    )

    labels = labels.reindex(
        columns=ACTIVITIES,
        fill_value=0
    )

    labels.columns = [
        f"label_{activity}"
        for activity in labels.columns
    ]

    labels_df = labels_df.merge(
        labels,
        left_on="user_id",
        right_index=True,
        how="left"
    )

    label_columns = [
        f"label_{activity}"
        for activity in ACTIVITIES
    ]

    labels_df[label_columns] = (
        labels_df[label_columns]
        .fillna(0)
        .astype("int8")
    )

    return labels_df    