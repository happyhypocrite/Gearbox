
import flowkit as fk
import pandas as pd
import os

def _parse_fcs_add_gate_label(wsp_path, wsp_fcs_dir, csv_dir):
    """ Uses the converted and processed .csv files and the .fcs files that they origin from. 
    Takes the Gate_Label from the .fcs file and applies it to the .csv """

    workspace = fk.Workspace(wsp_path, fcs_samples = wsp_fcs_dir)
    workspace.analyze_samples()
    sample_ids = list(workspace.get_sample_ids())
    print(f"Found {len(sample_ids)} samples to process")

    for sample_id in sample_ids:
        print(f"Processing {sample_id}")

        try:
            sample = workspace.get_sample(sample_id)
            gate_ids = workspace.get_gate_ids(sample_id)
            terminal_gates = []
            for gate_id in gate_ids:
                child_gates = workspace.get_child_gate_ids(sample_id, gate_id[0], gate_path= gate_id[1])
                if not child_gates:  # No children = terminal gate
                    terminal_gates.append(gate_id)
            
            # Get the number of events in the sample - each event needs a gate label
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

            sample_id_nofcs = sample_id.replace('.fcs','')
            csv_path = os.path.join(csv_dir, f"{sample_id_nofcs}.csv")
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

            output_path = os.path.join(csv_dir, f"{sample_id_nofcs}_with_gate_label.csv")
            csv_df.to_csv(output_path, index=False)
            print(f"Saved labeled CSV for {sample_id_nofcs}")

        except Exception as e:
            print(f"Error processing {sample_id}: {e}")
    print('.csv processing complete')
    return None

def _extract_gating_strategy(wsp_path, wsp_fcs_dir, output_path="./gating_structure.csv"):
    """Extract gating strategy from FlowJo workspace and save as a CSV for UNITO to use"""

    workspace = fk.Workspace(wsp_path, fcs_samples=wsp_fcs_dir)
    workspace.analyze_samples()
    
    # Get the first sample to analyze the gating tree structure
    sample_ids = workspace.get_sample_ids()
    if not sample_ids:
        print("No samples found in workspace")
        return None
    
    # Use first sample as reference for gating structure
    sample_id = list(sample_ids)[0]
    print(f"Using sample {sample_id} as reference for gating strategy")
    
    gating_data = []
    
    # Get all gates in the workspace
    gate_ids = workspace.get_gate_ids(sample_id)
    print(f"Found {len(gate_ids)} gates")
    
    for gate_id in gate_ids:
        try:
            gate_name, gate_path = gate_id
            
            # Get parent gate from gate_path - it's the last element in the path
            if len(gate_path) > 1:
                parent_gate = gate_path[-1]  # Last element is the direct parent
            else:
                parent_gate = None  # Root level gate has no parent
            
            # Get the gate object to extract dimensions
            try:
                gate = workspace.get_gate(sample_id, gate_name, gate_path)
                sample = workspace.get_sample(sample_id)
                
                # Try to get gate dimensions/channels
                x_axis = None
                y_axis = None
                
                # Method 1: Check if gate has dimensions attribute
                if hasattr(gate, 'dimensions'):
                    dims = gate.dimensions
                    if len(dims) >= 1:
                        dim_index = dims[0]
                        if isinstance(dim_index, int):
                            # If it's an integer, get marker name by index
                            channels = sample.channels
                            if dim_index < len(channels):
                                # Try to get the marker name (PnN or PnS parameter)
                                channel = channels[dim_index]
                                x_axis = (channel.get('pnn_label') or 
                                         channel.get('pns_label') or 
                                         channel.get('channel_name'))
                        else:
                            # If it's a Dimension object, try to find matching channel
                            dim_id = dim_index.id
                            try:
                                channel_idx = sample.get_channel_index(dim_id)
                                channel = sample.channels[channel_idx]
                                # Get the marker name, not the fluorochrome name
                                x_axis = (channel.get('pnn_label') or 
                                         channel.get('pns_label') or 
                                         channel.get('channel_name'))
                            except:
                                x_axis = dim_id  # Fallback
                    
                    if len(dims) >= 2:
                        dim_index = dims[1]
                        if isinstance(dim_index, int):
                            channels = sample.channels
                            if dim_index < len(channels):
                                channel = channels[dim_index]
                                y_axis = (channel.get('pnn_label') or 
                                         channel.get('pns_label') or 
                                         channel.get('channel_name'))
                        else:
                            dim_id = dim_index.id
                            try:
                                channel_idx = sample.get_channel_index(dim_id)
                                channel = sample.channels[channel_idx]
                                y_axis = (channel.get('pnn_label') or 
                                         channel.get('pns_label') or 
                                         channel.get('channel_name'))
                            except:
                                y_axis = dim_id  # Fallback
                
                # Method 2: Check for channel attributes (direct string values)
                elif hasattr(gate, 'x_channel'):
                    # Try to get marker name for x_channel
                    try:
                        channel_idx = sample.get_channel_index(gate.x_channel)
                        channel = sample.channels[channel_idx]
                        x_axis = (channel.get('pnn_label') or 
                                 channel.get('pns_label') or 
                                 gate.x_channel)
                    except:
                        x_axis = gate.x_channel
                    
                    if hasattr(gate, 'y_channel'):
                        try:
                            channel_idx = sample.get_channel_index(gate.y_channel)
                            channel = sample.channels[channel_idx]
                            y_axis = (channel.get('pnn_label') or 
                                     channel.get('pns_label') or 
                                     gate.y_channel)
                        except:
                            y_axis = gate.y_channel
                
                # Method 3: For range gates (1D gates)
                elif hasattr(gate, 'dimension'):
                    dim = gate.dimension
                    if isinstance(dim, int):
                        # Direct index
                        channels = sample.channels
                        if dim < len(channels):
                            channel = channels[dim]
                            x_axis = (channel.get('pnn_label') or 
                                     channel.get('pns_label') or 
                                     channel.get('channel_name'))
                    else:
                        # Dimension object
                        dim_id = dim.id
                        try:
                            channel_idx = sample.get_channel_index(dim_id)
                            channel = sample.channels[channel_idx]
                            x_axis = (channel.get('pnn_label') or 
                                     channel.get('pns_label') or 
                                     dim_id)
                        except:
                            x_axis = dim_id  # Fallback
                    y_axis = None
                
                print(f"Gate {gate_name} dimensions: X={x_axis}, Y={y_axis}")
                
            except Exception as gate_error:
                print(f"Could not get gate details for {gate_name}: {gate_error}")
                x_axis = None
                y_axis = None

            gating_data.append({
                'Gate': gate_name,
                'Parent_Gate': parent_gate,
                'X_axis': x_axis,
                'Y_axis': y_axis
            })
            
            print(f"Added gate: {gate_name}, Parent: {parent_gate}, X: {x_axis}, Y: {y_axis}")
            
        except Exception as e:
            print(f"Error processing gate {gate_id}: {e}")
            continue
    
    # Create DataFrame and save
    if gating_data:
        gating_df = pd.DataFrame(gating_data)
        
        # Sort by hierarchy - parents before children
        def sort_by_hierarchy(df):
            sorted_gates = []
            remaining = df.copy()
            processed_gates = set()
            
            # Add root gates first (no parent)
            root_gates = remaining[remaining['Parent_Gate'].isna()]
            sorted_gates.extend(root_gates.to_dict('records'))
            processed_gates.update(root_gates['Gate'].tolist())
            remaining = remaining.drop(root_gates.index)
            
            # Iteratively add gates whose parents have been processed
            while not remaining.empty:
                ready_gates = remaining[
                    remaining['Parent_Gate'].isin(processed_gates)
                ]
                
                if ready_gates.empty:
                    # No more gates can be processed - might be circular dependency
                    print("Warning: Possible circular dependency in gating hierarchy")
                    sorted_gates.extend(remaining.to_dict('records'))
                    break
                
                sorted_gates.extend(ready_gates.to_dict('records'))
                processed_gates.update(ready_gates['Gate'].tolist())
                remaining = remaining.drop(ready_gates.index)
            
            return pd.DataFrame(sorted_gates)
        
        gating_df = sort_by_hierarchy(gating_df)
        gating_df.to_csv(output_path, index=False)
        print(f"Gating strategy saved to {output_path}")
        print(f"Gating strategy:\n{gating_df}")
        return gating_df
    else:
        print("No gating data found")
        return None