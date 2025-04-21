import pandas as pd
from datetime import datetime
import logging
import os
from configuration import Config, Path
from CommonEnums import ErrorMessages

# Ensure the logs directory exists
if not os.path.exists('logs'):
    os.makedirs('logs')

# Use the logger without reconfiguring it
logger = logging.getLogger(__name__)

def cocomo_ii_effort_estimation(KLOC, A, B, EAF):
    """
    COCOMO II Effort Estimation Formula:
    Effort = A * (KLOC ** B) * EAF
    """
    return A * (KLOC ** B) * EAF

def apply_cocomo_model(df):
    try:
        logger.info("Starting COCOMO II effort estimation...")

        # Validate required columns
        required_columns = ['Object points']
        if not all(column in df.columns for column in required_columns):
            logger.error(f"Missing required columns: {required_columns}")
            raise ValueError(ErrorMessages.MISSING_REQUIRED_COLUMNS.value)

        # Convert Object Points to KLOC (adjust the conversion factor as needed)
        logger.info("Converting Object Points to KLOC...")
        df['KLOC'] = df['Object points'] * Config.KLOC_CONVERSION_FACTOR  # Use configuration value

        # Apply COCOMO II Effort Estimation Formula
        logger.info("Applying COCOMO II effort estimation formula...")
        A = Config.COCOMO_A  # Constant for organic projects
        B = Config.COCOMO_B  # Exponent for organic projects
        EAF = Config.COCOMO_EAF  # Effort Adjustment Factor (default is 1.0)

        df['COCOMO Effort'] = df.apply(lambda row: cocomo_ii_effort_estimation(row['KLOC'], A, B, EAF), axis=1)

        # Convert COCOMO Effort (person-months) to person-hours
        logger.info("Converting COCOMO Effort to person-hours...")
        df['COCOMO Effort (person-hours)'] = df['COCOMO Effort'] * Config.WORKING_DAYS_PER_MONTH * Config.WORKING_HOURS_PER_DAY

        # Save the updated dataset
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"cocomo_estimation_{timestamp}.xlsx"
        df.to_excel(output_file, index=False)
        logger.info(f"COCOMO-estimated dataset saved as {output_file}")

        return df

    except Exception as e:
        logger.error(f"An error occurred in apply_cocomo_model: {e}", exc_info=True)
        raise