# Databricks notebook source
aws_bucket_name = "databricks-dhdev-db-research"
mount_name = "databricks-dhdev-db-research"
#access_key = dbutils.secrets.get(scope = "aws", key = "XXXXXXX")
#secret_key = dbutils.secrets.get(scope = "aws", key = "XXXXXXX")
#encoded_secret_key = secret_key.replace("/", "%2F")
dbutils.fs.mount("s3a://%s" % aws_bucket_name, "/mnt/%s" % mount_name, extra_configs={
  "fs.s3a.credentialsType": "AssumeRole",
  "fs.s3a.stsAssumeRole.arn": "arn:aws:iam::415067613590:role/databricks-dhdev-db-s3-access"
})
#dbutils.fs.mount("s3a://%s:%s@%s" % (access_key, encoded_secret_key, aws_bucket_name), "/mnt/%s" % mount_name)
#dbutils.fs.put("/mnt/my_new_file", "This is a file in cloud storage.")
#display(dbutils.fs.ls("/mnt/%s" % mount_name))

# COMMAND ----------

mount_name = "databricks-dhdev-db-research"
dbutils.fs.unmount("/mnt/%s" % mount_name)

# COMMAND ----------

dbutils.fs.put("/mnt/databricks-dhdev-db-research/my_new_file", "This is a file in cloud storage.")

# COMMAND ----------

# MAGIC %sh ls /dbfs/mnt/databricks-dhdev-db-research

# COMMAND ----------

# MAGIC %sh cat /dbfs/mnt/databricks-dhdev-db-research/mount.err
