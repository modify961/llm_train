import json
import csv

# Path to your JSON file
json_file_path = 'data/data.json'

# CSV file output path
csv_file_path = 'D:/output.csv'

# Function to read JSON data and write to CSV
def json_to_csv(json_path, csv_path):
    # Read the JSON file
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)  # Load JSON data from file
    
    # Open the CSV file for writing
    with open(csv_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        
        # Write the header row
        writer.writerow(['category','human', 'assistant'])
        
        # Write data rows
        for entry in data:
            instruction = entry.get('instruction', '')  # Get 'instruction' or default to empty string
            output = entry.get('output', '')  # Get 'output' or default to empty string
            writer.writerow(['general',instruction, output])

# Execute the function
if __name__ == "__main__":
    json_to_csv(json_file_path, csv_file_path)
