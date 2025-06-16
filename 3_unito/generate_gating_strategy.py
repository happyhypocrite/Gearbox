import flowkit as fk
import pandas as pd
import os

def _parse_fcs_add_gate_label(wsp_path, wsp_fcs_dir, csv_dir):
    """ Uses the converted and processed .csv files and the .fcs files that they origin from. 
    Takes the Gate_Label from the .fcs file and applies it to the .csv """

    workspace = fk.Workspace(wsp_path, fcs_samples = wsp_fcs_dir)
    workspace.analyze_samples()
    for sample_id in workspace.get_sample_ids():
        print(f"Processing {sample_id}")

        try:
            sample = workspace.get_sample(sample_id)
            gate_ids = workspace.get_gate_ids(sample_id)
            terminal_gates = []
            for gate_id in gate_ids:
                child_gates = workspace.get_child_gate_ids(sample_id, gate_id[0], gate_path= gate_id[1])
                if not child_gates:  # No children = terminal gate
                    terminal_gates.append(gate_id)
            
            # Get the number of events in the sample
            num_events = len(sample.get_events(source= 'raw'))
            
            # Initialize gate labels with "Ungated"
            gate_labels = ["Ungated"] * num_events

            for gate_id in terminal_gates:
                try:
                    membership = workspace.get_gate_membership(sample_id, gate_id[0], gate_path= gate_id[1])
                    # Update labels for cells in this terminal gate
                    for i, is_member in enumerate(membership):
                        if is_member:
                            gate_labels[i] = gate_id
                except Exception as e:
                    print(f"Error getting membership for terminal gate {gate_id}: {e}")

            gate_df = pd.DataFrame({'Gate_Label': gate_labels})

            csv_path = os.path.join(csv_dir, f"{sample_id}.csv")
            if not os.path.exists(csv_path):
                print(f"CSV for {sample_id} not found, skipping.")
                continue

            csv_df = pd.read_csv(csv_path)

            if len(csv_df) != len(gate_df):
                print(f"Mismatch in cell count for {sample_id}: CSV has {len(csv_df)}, FCS has {len(gate_df)}")
                continue

            # Add the Gate_Label to your CSV
            csv_df["Gate_Label"] = gate_df["Gate_Label"].values

            # Save output
            sample_id_nofcs = sample_id.replace('.fcs','')
            output_path = os.path.join(csv_dir, f"{sample_id_nofcs}_with_gate_label.csv")
            csv_df.to_csv(output_path, index=False)
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
    gate_ids = workspace.get_gate_ids(sample)
    
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