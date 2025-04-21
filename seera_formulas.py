import pandas as pd
import logging
from configuration import Config
from CommonEnums import ErrorMessages

# Use the logger without reconfiguring it
logger = logging.getLogger(__name__)

def calculate_object_points(df):
    try:
        logger.info("Calculating Object Points...")
        if 'Object points' in df.columns:
            logger.info("Object Points already exist in the dataset. Skipping calculation.")
            return df  # If Object Points already exist, use them

        if 'Estimated size' in df.columns:
            df['Object Points'] = df['Estimated size'] * Config.OBJECT_POINTS_SCALING_FACTOR
            logger.info("Object Points calculated using 'Estimated size'.")
        elif 'Other sizing method' in df.columns:
            df['Object Points'] = df['Other sizing method'] * Config.OBJECT_POINTS_SCALING_FACTOR
            logger.info("Object Points calculated using 'Other sizing method'.")
        else:
            logger.warning("Cannot calculate Object Points. No relevant attributes found.")
        
        return df
    except Exception as e:
        logger.error(f"Error in calculate_object_points: {e}", exc_info=True)
        raise

def calculate_estimated_effort(df):
    try:
        logger.info("Calculating Estimated Effort...")
        required_columns = ['Estimated duration', 'Dedicated team members', 'Team size', 'Daily working hours']
        if not all(col in df.columns for col in required_columns):
            logger.warning(f"Required columns for Estimated Effort calculation not found: {required_columns}. Skipping.")
            return df

        df['Estimated effort'] = (
            df['Estimated duration'] * 
            (df['Dedicated team members'] + (df['Team size'] - df['Dedicated team members']) * Config.PART_TIME_FACTOR) * 
            (df['Daily working hours'] * Config.WORKING_DAYS_PER_MONTH)
        )
        logger.info("Estimated Effort calculated successfully.")
        return df
    except Exception as e:
        logger.error(f"Error in calculate_estimated_effort: {e}", exc_info=True)
        raise

def calculate_actual_effort(df):
    try:
        logger.info("Calculating Actual Effort...")
        required_columns = ['Actual duration', 'Dedicated team members', 'Team size', 'Daily working hours']
        if not all(col in df.columns for col in required_columns):
            logger.warning(f"Required columns for Actual Effort calculation not found: {required_columns}. Skipping.")
            return df

        df['Actual effort'] = (
            df['Actual duration'] * 
            (df['Dedicated team members'] + (df['Team size'] - df['Dedicated team members']) * Config.PART_TIME_FACTOR) * 
            (df['Daily working hours'] * Config.WORKING_DAYS_PER_MONTH)
        )
        logger.info("Actual Effort calculated successfully.")
        return df
    except Exception as e:
        logger.error(f"Error in calculate_actual_effort: {e}", exc_info=True)
        raise

def calculate_project_gain_loss(df):
    try:
        logger.info("Calculating Project Gain/Loss...")
        required_columns = ['Contract price', 'Actual incurred costs']
        if not all(col in df.columns for col in required_columns):
            logger.warning(f"Required columns for Project Gain/Loss calculation not found: {required_columns}. Skipping.")
            return df

        df['% project gain (loss)'] = (
            (df['Contract price'] - df['Actual incurred costs']) / df['Contract price'] * 100
        )
        logger.info("Project Gain/Loss calculated successfully.")
        return df
    except Exception as e:
        logger.error(f"Error in calculate_project_gain_loss: {e}", exc_info=True)
        raise

def apply_seera_formulas(df):
    try:
        logger.info("Applying SEERA formulas...")
        df = calculate_object_points(df)
        df = calculate_estimated_effort(df)
        df = calculate_actual_effort(df)
        df = calculate_project_gain_loss(df)
        logger.info("SEERA formulas applied successfully.")
        return df
    except Exception as e:
        logger.error(f"Error in apply_seera_formulas: {e}", exc_info=True)
        raise