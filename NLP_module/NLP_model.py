import pandas as pd
import os

component = pd.read_excel("dataset/smells.xlsx")
component = component['ComponentName']

path = "../projects"

# Crea una lista vuota per salvare i nomi dei progetti
projects = []

# Itera su tutti i file e le cartelle nella cartella specificata
for item in os.listdir(path):
    # Se l'elemento nella cartella è una cartella, aggiungi il nome alla lista
    if os.path.isdir(os.path.join(path, item)):
        projects.append(item)

# Stampa la lista di nomi dei progetti
print(projects)
possible_subfolders = ["src/java", "src/main", "src/main/java", "src"]
classes = []
for k in range(len(projects)):
    #print(projects[k])
    for i in range(len(component)):
        line = component[i].strip()
        line = line.replace(".","/") + ".java"
        for subfolder in possible_subfolders:
            full_path = os.path.join(path, projects[k], subfolder, line)
            if os.path.exists(full_path):
                with open(full_path, "r") as f:
                    contenuto = f.read()
                classes.append(contenuto)
                break # Esci dal ciclo for se hai trovato il file
print(len(classes))
