from neo4j import GraphDatabase
from dotenv import load_dotenv
import os

load_dotenv()

# Get credentials
uri = os.getenv("NEO4J_URI")
user = os.getenv("NEO4J_USER") 
password = os.getenv("NEO4J_PASSWORD")

print(f"Testing Neo4j Aura...")
print(f"URI: {uri}")

try:
    # Connect
    driver = GraphDatabase.driver(uri, auth=(user, password))
    driver.verify_connectivity()
    
    # Run test query
    with driver.session() as session:
        result = session.run("RETURN 'Connected!' as msg")
        msg = result.single()["msg"]
        print(f"\n✅ {msg}")
        print("✅ Neo4j Aura is working!")
    
    driver.close()
    
except Exception as e:
    print(f"\n❌ Connection failed: {e}")
    print("\nDouble-check:")
    print("1. Database status is 'Running' in console")
    print("2. URI starts with 'neo4j+s://'")
    print("3. Password is correct")
