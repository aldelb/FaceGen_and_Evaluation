
def write_params(f, title, params):
    f.write(f"# --- {title}\n")
    for argument in params.keys() :
        f.write(f"{argument} : {params[argument]}\n\n")

def save_params(saved_path, model, D = None):


    file_path = saved_path + "parameters.txt"
    f = open(file_path, "w")

    f.write("-"*10 + "Models" + "-"*10 + "\n")
    f.write("-"*10 + "Generateur" + "-"*10 + "\n")
    f.write(str(model))

    f.close()