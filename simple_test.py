import pandas as pd
from datetime import datetime

# Simple test cases with additional factors
test_cases = [
    {
        'Object_points': 100,
        'Team_size': 4,
        'Duration': 5,
        'Daily_hours': 8,
        'Development_type': 'New',  # New/Enhancement/Maintenance
        'Dedicated_team_members': 3,  # Full-time team members
        'Team_experience': 'Medium',  # Low/Medium/High
        'Project_complexity': 'Medium'  # Low/Medium/High
    },
    {
        'Object_points': 200,
        'Team_size': 6,
        'Duration': 8,
        'Daily_hours': 8,
        'Development_type': 'Enhancement',
        'Dedicated_team_members': 5,
        'Team_experience': 'High',
        'Project_complexity': 'High'
    },
    {
        'Object_points': 350,
        'Team_size': 10,
        'Duration': 12,
        'Daily_hours': 8,
        'Development_type': 'New',
        'Dedicated_team_members': 8,
        'Team_experience': 'Medium',
        'Project_complexity': 'High'
    }
]

def calculate_complexity_score(case):
    # Base complexity weights
    complexity_weights = {
        'Low': 0.8,
        'Medium': 1.0,
        'High': 1.2
    }
    
    # Development type weights
    dev_type_weights = {
        'New': 1.2,
        'Enhancement': 1.0,
        'Maintenance': 0.8
    }
    
    # Calculate size-based complexity
    size_factor = case['Object_points'] / 100  # Normalize by 100 points
    
    # Calculate team-based complexity
    team_ratio = case['Dedicated_team_members'] / case['Team_size']
    team_factor = 1 + (1 - team_ratio) * 0.2  # More part-time members increase complexity
    
    # Get weights from the mappings
    complexity_weight = complexity_weights[case['Project_complexity']]
    dev_weight = dev_type_weights[case['Development_type']]
    
    # Calculate final complexity score
    complexity_score = (
        size_factor * 
        team_factor * 
        complexity_weight * 
        dev_weight
    )
    
    return round(complexity_score, 2)

def calculate_team_productivity(case):
    # Experience weights
    experience_weights = {
        'Low': 0.8,
        'Medium': 1.0,
        'High': 1.2
    }
    
    # Calculate effective team size
    effective_team = (
        case['Dedicated_team_members'] + 
        (case['Team_size'] - case['Dedicated_team_members']) * 0.5  # Part-time factor
    )
    
    # Base productivity (hours per person per month)
    base_productivity = case['Daily_hours'] * 22  # 22 working days
    
    # Adjust for team experience
    experience_factor = experience_weights[case['Team_experience']]
    
    # Calculate final productivity
    productivity = base_productivity * effective_team * experience_factor
    
    return round(productivity, 2)

def run_predictions():
    # Print test case details
    print("\nTest Cases:")
    print("-" * 50)
    for i, case in enumerate(test_cases, 1):
        print(f"\nCase {i}:")
        for key, value in case.items():
            print(f"{key}: {value}")
    
    # Make predictions
    predictions = []
    for case in test_cases:
        # Calculate complexity score and team productivity
        complexity_score = calculate_complexity_score(case)
        team_productivity = calculate_team_productivity(case)
        
        pred = {
            'Object_Points': case['Object_points'],
            'Team_Size': case['Team_size'],
            'Duration': case['Duration'],
            'Daily_Hours': case['Daily_hours'],
            'Development_Type': case['Development_type'],
            'Team_Experience': case['Team_experience'],
            'Project_Complexity': case['Project_complexity'],
            'Complexity_Score': complexity_score,
            'Team_Productivity': team_productivity
        }
        
        # Calculate predictions for each model
        # COCOMO Estimate (adjusted with complexity)
        kloc = case['Object_points'] * 0.2  # Convert Object Points to KLOC
        pred['COCOMO_Estimate'] = 2.4 * (kloc ** 1.05) * complexity_score  # person-months
        pred['COCOMO_Estimate'] *= 132  # Convert to hours (22 days * 6 hours)
        
        # Dynamic Estimate (based on team productivity)
        pred['Dynamic_Estimate'] = (
            case['Duration'] * 
            team_productivity
        )
        
        # Hybrid Estimate (weighted combination with complexity adjustment)
        base_hybrid = (
            pred['COCOMO_Estimate'] * 0.4 + 
            pred['Dynamic_Estimate'] * 0.6
        )
        
        # Adjust hybrid estimate based on complexity and team experience
        experience_adj = 1.0
        if case['Team_experience'] == 'High':
            experience_adj = 0.9
        elif case['Team_experience'] == 'Low':
            experience_adj = 1.2
            
        pred['Hybrid_Estimate'] = base_hybrid * experience_adj
        
        # Round all estimates to 2 decimal places
        for key in ['COCOMO_Estimate', 'Dynamic_Estimate', 'Hybrid_Estimate']:
            pred[key] = round(pred[key], 2)
        
        predictions.append(pred)
    
    # Create results DataFrame
    results = pd.DataFrame(predictions)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"test_predictions_{timestamp}.xlsx"
    results.to_excel(output_file, index=False)
    
    # Display results
    print("\nPrediction Results:")
    print("-" * 50)
    print(results)
    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    run_predictions() 