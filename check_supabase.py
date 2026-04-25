from dotenv import load_dotenv, find_dotenv
import os
from supabase import create_client

load_dotenv(find_dotenv())

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

print("URL set:", bool(url))
print("Service key set:", bool(key))

supabase = create_client(url, key)
print("Client initialized OK")

resp = supabase.table("documents").select("id").limit(1).execute()
print("Table check OK, rows:", len(resp.data) if resp.data else 0)