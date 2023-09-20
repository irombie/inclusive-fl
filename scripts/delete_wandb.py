from datetime import datetime, timedelta

# import wandb

# # Define the cutoff date (1 month ago)
# cutoff_date = datetime.now() - timedelta(days=30)

# # List all projects
# projects = wandb.api.list_projects()

# # Loop through projects and delete those that are older than the cutoff date
# for project in projects:
#     project_id = project["id"]
#     project_name = project["name"]
#     last_updated = project["updated_at"]

#     # Convert last_updated to a datetime object
#     last_updated_date = datetime.strptime(last_updated, "%Y-%m-%dT%H:%M:%S.%fZ")

#     if last_updated_date < cutoff_date:
#         print(f"Deleting project: {project_name}")
#         wandb.api.delete(f"/projects/{project_id}")
