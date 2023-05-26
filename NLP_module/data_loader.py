import pandas as pd
import os
import git

def load_data():
    component = pd.read_csv("dataset/complete_dataset.csv")
    #component = component.drop(['Project','Version','Smell'],axis=1)
    df = pd.DataFrame()

    path = "../projects"
    projects = []

    # Itera su tutti i file e le cartelle nella cartella specificata
    for item in os.listdir(path):
        # Se l'elemento nella cartella è una cartella, aggiungi il nome alla lista
        if os.path.isdir(os.path.join(path, item)):
            projects.append(item)
    # Stampa la lista di nomi dei progetti
    print(projects)

    possible_subfolders = ["src/java", "src/main", "src/main/java", "src"]
    #classes = []

    for k in range(len(projects)):
        print(projects[k])
        for i in range(len(component)):
            line = component.loc[i,'ComponentName'].strip()
            line = line.replace(".","/") + ".java"
            for subfolder in possible_subfolders:
                full_path = os.path.join(path, projects[k], subfolder, line)
                if os.path.exists(full_path):
                    # with open(full_path, "r") as f:
                    #     contenuto = f.read()
                    #classes.append(contenuto)
                    temp_df = pd.DataFrame({
                        'Project_name': projects[k], 
                        'Component_name': [component.loc[i, 'ComponentName']],
                        'CBO' :[component.loc[i,'CBO']],
                        'CYCLO' :[component.loc[i,'CYCLO']],
                        'DIT' :[component.loc[i,'DIT']],
                        'ELOC' :[component.loc[i,'ELOC']],
                        'FanIn' :[component.loc[i,'FanIn']],
                        'FanOut' :[component.loc[i,'FanOut']],
                        'LCOM' :[component.loc[i,'LCOM']],
                        'LOC' :[component.loc[i,'LOC']],
                        'LOCNAMM' :[component.loc[i,'LOCNAMM']],
                        'NOA' :[component.loc[i,'NOA']],
                        'NOC' :[component.loc[i,'NOC']],
                        'NOM' :[component.loc[i,'NOM']],
                        'NOMNAMM' :[component.loc[i,'NOMNAMM']],
                        'NOPA' :[component.loc[i,'NOPA']],
                        'PMMM' :[component.loc[i,'PMMM']],
                        'PRB' :[component.loc[i,'PRB']],
                        'WLOCNAMM' :[component.loc[i,'WLOCNAMM']],
                        'WMC' :[component.loc[i,'WMC']],
                        'NOM' :[component.loc[i,'NOM']],
                        'WMCNAMM': [component.loc[i,'WMCNAMM']],
                        'NMNOPARAM': [component.loc[i,'NMNOPARAM']],
                        'CDSBP': [component.loc[i,'CDSBP']],
                        'CC': [component.loc[i,'CC']],
                        'LC': [component.loc[i,'LC']],
                        'LZC': [component.loc[i,'LZC']],
                        'RB': [component.loc[i,'RB']],
                        'SC': [component.loc[i,'SC']]
                    })
                    df = pd.concat([df, temp_df], ignore_index=True)
                    break # Esci dal ciclo for se hai trovato il file

    #print(len(classes))
    #print(len(df))
    #print(df)
    # final_df = pd.DataFrame()
    # final_df['component'] = df['Component']
    # final_df['labels'] = df.iloc[:, 1:].values.tolist()

    return df

f_df = load_data()
f_df.to_csv("dataset/final_dataset.csv", index=False)

