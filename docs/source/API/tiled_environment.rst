tiled\_environment
==================

.. automodule:: tiled_environment

.. autoclass:: TiledEnvConfig

.. autoclass:: Tile
   :members: __init__, build, search, 
   
.. autoclass:: Layer
   :members: n_tiles_per_action, __init__, __len__, build_tiles, get_global_tile_index, _do_build_tile, _do_build_three_columns  
   
   
.. autoclass:: Tiles
   :members: __init__, __getitem__, __len__, build
   
   
.. autoclass:: TiledEnv
   :members: from_options, __init__, action_space, n_actions, n_states, config, step, reset, get_state_action_tile_matrix, get_action, save_current_dataset, create_tiles, get_aggregated_state, initialize_column_counts, all_columns_visited, initialize_distances, apply_action. total_current_distortion, featurize_state_action, featurize_raw_state, _create_column_scales, _validate
   
   

   
   
   

   
   
   .. rubric:: Classes

   .. autosummary::
   
      TiledEnv
      TiledEnvConfig
   
   

   
   
   



