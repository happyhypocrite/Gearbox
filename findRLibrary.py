import subprocess 
import os
import glob

def find_rscript():
    possible_directories = [
        r"C:\Users\mfbx2rdb\AppData\Local\Programs\R\R-4.4.2"
        #r"C:\Program Files\R",                   
        #r"C:\Program Files (x86)\R",
        #os.environ.get("ProgramFiles"),  
        #os.environ.get("ProgramFiles(x86)")
    ]
    for directory in possible_directories:
        if directory:
            rscript_paths = glob.glob(os.path.join(directory, "**", "Rscript.exe"), recursive=True)
            if rscript_paths:
                return rscript_paths[0]
    return None

def find_r_library_locations(rscript_path):
    try:
        command = [rscript_path, "-e", "cat(paste(.libPaths(), collapse='\n'))"]
        print(f"Executing command: {' '.join(command)}")
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        library_locations = result.stdout.split('\n')
        return library_locations
    except subprocess.CalledProcessError as e:
        print(f"Error finding R library locations: {e}")
        print(f"stderr: {e.stderr}")
        return None

# Example usage
rscript_path = find_rscript()
if rscript_path:
    library_locations = find_r_library_locations(rscript_path)
    if library_locations:
        print("R library locations:")
        for location in library_locations:
            print(location)
    else:
        print("No library locations found or an error occurred.")
else:
    print("Rscript.exe not found.")