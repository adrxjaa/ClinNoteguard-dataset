import pandas as pd
import os

base_dir = r'c:\Users\rajee\Downloads\dataset'
challenge_dir = os.path.join(base_dir, 'aci-bench-corpus', 'challenge_data')

files = {
    'train': 'train.csv',
    'valid': 'valid.csv',
    'test1': 'clinicalnlp_taskB_test1.csv',
    'test2': 'clinicalnlp_taskC_test2.csv',
    'test3': 'clef_taskC_test3.csv'
}

patterns = {
    'train': ['Member 1', 'Member 2', 'Member 3'],
    'valid': ['Member 1', 'Member 2', 'Member 3'],
    'test1': ['Member 3', 'Member 1', 'Member 2'],
    'test2': ['Member 2', 'Member 3', 'Member 1'],
    'test3': ['Member 3', 'Member 1', 'Member 2']
}

all_assignments = []
unique_encounters = set()

for split, filename in files.items():
    df = pd.read_csv(os.path.join(challenge_dir, filename))
    pattern = patterns[split]
    
    for i, row in df.iterrows():
        enc_id = row['encounter_id']
        unique_encounters.add(enc_id)
        
        member = pattern[i % 3]
        
        # Determine output file
        member_num = member.split()[-1]
        output_file = f"ehr_member{member_num}_{split}.csv"
        
        all_assignments.append({
            'member': member,
            'split': split,
            'encounter_id': enc_id,
            'source_file': filename,
            'output_file': output_file
        })

df_all = pd.DataFrame(all_assignments)

# Verify counts
print("Total unique encounters:", len(unique_encounters))

summary = df_all.groupby(['split', 'member']).size().unstack(fill_value=0)
summary = summary[['Member 1', 'Member 2', 'Member 3']] # Reorder columns
# Reorder rows to match Train, Valid, Test1, Test2, Test3
summary = summary.reindex(['train', 'valid', 'test1', 'test2', 'test3'])
summary.loc['TOTAL'] = summary.sum()
print("\nAssignment Summary:\n", summary)

# Save main assignment file
df_all.to_csv(os.path.join(base_dir, 'aci_group_assignment.csv'), index=False)

# Save member specific assignment files
for m in ['Member 1', 'Member 2', 'Member 3']:
    member_num = m.split()[-1]
    df_m = df_all[df_all['member'] == m].copy()
    # Member specific file columns: split, encounter_id, source_file, output_file
    df_m = df_m[['split', 'encounter_id', 'source_file', 'output_file']]
    df_m.to_csv(os.path.join(base_dir, f'member{member_num}_assignment.csv'), index=False)

print("\nFiles generated successfully.")
