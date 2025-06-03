import flowkit as fk
import pandas as pd
import os

def _parse_fcs_add_gate_label(wsp_path, wsp_fcs_dir, csv_dir):
    """ Uses the converted and processed .csv files and the .fcs files that they origin from. 
    Takes the Gate_Label from the .fcs file and applies it to the .csv """

    workspace = fk.Workspace(wsp_path, fcs_samples = wsp_fcs_dir)
    for sample_id in workspace.get_sample_ids():
        print(f"Processing {sample_id}")

        try:
            sample = workspace.get_sample(sample_id)
            workspace.apply_gates(sample)

            # Get gate membership: a DataFrame of bools (True if the cell is in a gate)
            gate_df = sample.get_gate_membership()
            # Combine to make a Gate_Label column
            def resolve_gate(row):
                for gate in gate_df.columns:
                    if row[gate]:
                        return gate  # Return the first gate the cell belongs to
                return "Other"

            gate_df["Gate_Label"] = gate_df.apply(resolve_gate, axis=1)

            csv_path = os.path.join(csv_dir, f"{sample_id}.csv")
            if not os.path.exists(csv_path):
                print(f"CSV for {sample_id} not found, skipping.")
                continue

            csv_df = pd.read_csv(csv_path)

            if len(csv_df) != len(gate_df):
                print(f"Mismatch in cell count for {sample_id}, skipping.")
                continue

            # Add the Gate_Label to your CSV
            csv_df["Gate_Label"] = gate_df["Gate_Label"].values

            # Save output
            csv_df.to_csv(os.path.join(csv_dir, f"{sample_id}_with_labels.csv"), index=False)
            print(f"Saved labeled CSV for {sample_id}")

        except Exception as e:
            print(f"Error processing {sample_id}: {e}")
    
    return None




def _extract_gating_strategy(wsp_path, wsp_fcs_dir, output_path="./gating_structure.csv"):
    """Extract gating strategy from FlowJo workspace and save as CSV"""

    workspace = fk.Workspace(wsp_path, fcs_samples = wsp_fcs_dir)
    # Get the first sample to analyze the gating tree structure
    sample_ids = workspace.get_sample_ids()
    if not sample_ids:
        print("No samples found in workspace")
        return None
    
    # Use first sample as reference for gating structure
    sample = workspace.get_sample(sample_ids[0])
    
    gating_data = []
    
    # Get all gates in the workspace
    gate_ids = workspace.get_gate_ids()
    
    for gate_id in gate_ids:
        try:
            gate = workspace.get_gate(gate_id)
            
            # Get gate properties
            gate_name = gate_id
            
            # Get parent gate (if any)
            parent_gates = workspace.get_parent_gate_ids(gate_id)
            parent_gate = parent_gates[0] if parent_gates else None
            
            # Get the dimensions (axes) for this gate
            dimensions = gate.get_dimensions()
            
            # Extract X and Y axes
            x_axis = dimensions[0] if len(dimensions) > 0 else None
            y_axis = dimensions[1] if len(dimensions) > 1 else None
            
            gating_data.append({
                'Gate': gate_name,
                'Parent_Gate': parent_gate,
                'X_axis': x_axis,
                'Y_axis': y_axis
            })
            
        except Exception as e:
            print(f"Error processing gate {gate_id}: {e}")
            continue
    
    # Create DataFrame and save
    if gating_data:
        gating_df = pd.DataFrame(gating_data)
        
        # Sort by hierarchy (gates with no parent first, then by dependency)
        def sort_by_hierarchy(df):
            sorted_gates = []
            remaining = df.copy()
            
            while not remaining.empty:
                # Find gates whose parents are already processed or have no parent
                ready_gates = remaining[
                    (remaining['Parent_Gate'].isna()) | 
                    (remaining['Parent_Gate'].isin([g['Gate'] for g in sorted_gates]))
                ]
                
                if ready_gates.empty:
                    # If no gates are ready, there might be a circular dependency
                    # Add remaining gates anyway
                    sorted_gates.extend(remaining.to_dict('records'))
                    break
                
                sorted_gates.extend(ready_gates.to_dict('records'))
                remaining = remaining.drop(ready_gates.index)
            
            return pd.DataFrame(sorted_gates)
        
        gating_df = sort_by_hierarchy(gating_df)
        gating_df.to_csv(output_path, index=False)
        print(f"Gating strategy saved to {output_path}")
        return gating_df
    
    return None