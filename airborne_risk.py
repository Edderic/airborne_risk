import numpy as np


CO2_GENERATION_RATE_MAPPING = [
    {
        '1 to <3': {'coef': 0.00279, 'intercept': -0.000017},
        '3 to <6': {'coef': 0.002774, 'intercept': -0.000016},
        '6 to <11': {'coef': 0.002749, 'intercept': -0.000015},
        '11 to <16': {'coef': 0.002713, 'intercept': -0.000015},
        '16 to <21': {'coef': 0.002698, 'intercept': -0.000014},
        '21 to <30': {'coef': 0.002688, 'intercept': -0.000013},
        '30 to <40': {'coef': 0.002696, 'intercept': -0.000014},
        '40 to <50': {'coef': 0.002693, 'intercept': -0.000012},
        '50 to <60': {'coef': 0.002694, 'intercept': -0.000014},
        '60 to <70': {'coef': 0.002716, 'intercept': -0.000014},
        '70 to <80': {'coef': 0.002722, 'intercept': -0.000015},
        '<1': {'coef': 0.002813, 'intercept': -0.000015},
        '>=80': {'coef': 0.002729, 'intercept': -0.000015}
        },
    {
        '1 to <3': {'coef': 0.001459, 'intercept': 0.000055},
        '11 to <16': {'coef': 0.003397, 'intercept': 0.00001},
        '16 to <21': {'coef': 0.00376, 'intercept': -0.000012},
        '21 to <30': {'coef': 0.004014, 'intercept': -0.000043},
        '3 to <6': {'coef': 0.001878, 'intercept': 0.000019},
        '30 to <40': {'coef': 0.00381, 'intercept': -0.000029},
        '40 to <50': {'coef': 0.003891, 'intercept': -0.000064},
        '50 to <60': {'coef': 0.003863, 'intercept': -0.000022},
        '6 to <11': {'coef': 0.0025, 'intercept': 0.0},
        '60 to <70': {'coef': 0.003323, 'intercept': -0.000028},
        '70 to <80': {'coef': 0.003177, 'intercept': -0.000003},
        '<1': {'coef': 0.000897, 'intercept': 0.00001},
        '>=80': {'coef': 0.003, 'intercept': 0.0}
    }
]

INFECTOR_ACTIVITY_TO_FACTOR = {
  "Resting – Oral breathing": 1,
  "Resting – Speaking": 4.7,
  "Resting – Loudly speaking": 30.3,
  "Standing – Oral breathing": 1.2,
  "Standing – Speaking": 5.7,
  "Standing – Loudly speaking": 32.6,
  "Light exercise – Oral breathing": 2.8,
  "Light exercise – Speaking": 13.2,
  "Light exercise – Loudly speaking": 85,
  "Heavy exercise – Oral breathing": 6.8,
  "Heavy exercise – Speaking": 31.6,
  "Heavy exercise – Loudly speaking": 204,
}

LITERS_PER_SECOND_TO_CUBIC_METERS_PER_HOUR = 3.6


class UserError(BaseException):
    pass

def activity_to_met(activity):
    activities_to_met = {
        "Calisthenics—light effort": 2.8,
        "Calisthenics—moderate effort": 3.8,
        "Calisthenics—vigorous effort": 8.0,
        "Child care": 2.5, # 2 - 3
        "Cleaning, sweeping—moderate effort": 3.8,
        "Custodial work—light": 2.3,
        "Dancing—aerobic, general": 7.3,
        "Dancing—general": 7.8,
        "Health club exercise classes—general": 5.0,
        "Kitchen activity—moderate effort": 3.3,
        "Lying or sitting quietly": 1.0,
        "Sitting reading, writing, typing": 1.3,
        "Sitting at sporting event as spectator": 1.5,
        "Sitting tasks, light effort (e.g, office work)": 1.5,
        "Sitting quietly in religious service": 1.3,
        "Sleeping": 0.95,
        "Standing quietly": 1.3,
        "Standing tasks, light effort (e.g, store clerk, filing)": 3.0,
        "Walking, less than 2 mph, level surface, very slow": 2.0,
        "Walking, 2.8 mph to 3.2 mph, level surface, moderate pace": 3.5,
    }

    return activities_to_met[activity]

def wastewater_to_proba_infectious(
    wastewater_level,
    wastewater_level_at_peak_omicron = 11500,
    proportion_infected_at_peak_omicron = 0.25
):
    """
    Converts wastewater levels to cases.

    Parameters:
        wastewater_level: int or float
            RNA copies / mL

        wastewater_level_at_peak_omicron: int or float
            The highest amount of RNA copies / mL recorded.

            Defaults to 11500 (which is the highest in Boston wastewater, North)

        proportion_infected_at_peak_omicron: float
            The proportion of people infected during the first Omicron wave.

            Defaults to 0.25. This is just an assumption.
    """

    return proportion_infected_at_peak_omicron * wastewater_level / wastewater_level_at_peak_omicron

def average_event_risk_tolerance(
    risk_budget=0.017552,
    number_of_events=24,
):
    """
    Assuming each event has the same amount of risk, how risky does each event have to be
    to reach the desired risk budget.

    Parameters:
        risk_budget: float
            The total risk allowable

        number_of_events: int
            The number of events in which to evenly divide the risk_budget.
    """
    return 1 - (1 - risk_budget) ** (1 / number_of_events)


def compute_risk_assuming_infector_is_present(
    quanta_per_hour=3.3 * 18.6,
    cadr_m3_per_hour=100,
    inhalation_rate_m3_per_hour=0.28,
    susceptible_mask_exposure_reduction_factor=1,
    infector_mask_exposure_reduction_factor=1,
    inhalation_factor=1,
    exhalation_activity_factor=1,
    time_hours=1

):
    """
    Long-range airborne transmission model based on Jimenez.

    Parameters:
        quanta_per_hour: int
            How infectious the infector is. The higher the more infectious.

            Defaults to 3.3 * 18.6, which is Jimenez et. al.'s estimate for Omicron BA.2

        cadr_m3_per_hour: float
            The clean air delivery rate (CADR) in terms of cubic meters per
            hour. Typically the sum of ventilation rate and filtration rate.

        inhalation_rate_m3_per_hour: float
            The basic inhalation rate in cubic meters per hour.

            Defaults to 0.28

        susceptible_mask_exposure_reduction_factor: float
            The mask exposure reduction factor for the susceptible.

            Defaults to 1 (i.e. no mask)

        infector_mask_exposure_reduction_factor: float
            The mask exposure reduction factor for the infector.

            Defaults to 1 (i.e. no mask)

        inhalation_factor: float
            The faster a susceptible inhales, the faster the accumulation of quanta.

            Defaults to 1

        exhalation_activity_factor: float
            The faster a infectious individual exhales, the faster the generation rate of quanta.

            Defaults to 1

        time_hours: float
            The number of hours the susceptible is spending in the room.

            Defaults to 1

    """
    return 1 - np.exp(-quanta_per_hour / cadr_m3_per_hour  \
        * inhalation_rate_m3_per_hour
        / susceptible_mask_exposure_reduction_factor \
        / infector_mask_exposure_reduction_factor \
        * inhalation_factor * exhalation_activity_factor \
        * time_hours)


def covid_risk_budget_within_months(
    num_months=6,
    lifespan_left_in_terms_of_num_months=120,
    long_covid_risk_tolerance_within_remaining_lifespan=0.1,
    long_covid_risk_per_infection=0.05
):
    """
    What is the recommended total risk tolerance of getting infected with COVID within the number of months?

    Assumptions: People can get COVID once every 6 months.

    Parameters:
        num_months: int
            The span of time that one can get COVID before being susceptible to getting COVID again.
            Defaults to 6.

        lifespan_left_in_terms_of_num_months: int
            This is essentially the number of years left / (number of months / number of years)
            Defaults to 120.

        long_covid_risk_tolerance_within_remaining_lifespan: float
            The amount of risk one is willing to take in getting Long COVID within remaining lifespan.
            Defaults to 10%

        long_covid_risk_per_infection: float
            The risk of getting Long COVID per infection.
            Defaults to 5%
    """
    return (1 - (1 - long_covid_risk_tolerance_within_remaining_lifespan)**(1/120)) / long_covid_risk_per_infection


def getCO2GenerationRate(met, man, age):
    """
    Meant for extrapolating CO2 generation rate given met, sex, and age
      Params:
       met: a number
         Higher met, higher CO2 breathed out
       man: boolean
         True if man, False, otherwise.
       age: string
         Age groups
     Returns:
       CO2 generation rate (L/s)
    """


    model = CO2_GENERATION_RATE_MAPPING[man * 1][age]

    return model['coef'] * met + model['intercept']


def average_m_f_co2_generation_rate(met, age='30 to <40', occupancy=1):

    """
    What is the average CO2 generation rate?

    Parameters:
        met: float
            The higher the met, the more intensive the activity is.
        age: string
            Range for age
        occupancy: int
            The number of people

    Returns generation rate in m3/h
    """
    return (getCO2GenerationRate(met, man=0, age=age) + getCO2GenerationRate(met, man=1, age=age)) / 2 \
        * LITERS_PER_SECOND_TO_CUBIC_METERS_PER_HOUR * occupancy


def probability_that_someone_is_infectious_in_room(proba_infectious, occupancy=None):
    """
    Assuming that each individual has the same probability of being infectious,
    what is the probability that at least one person is infectious in the room?

    Parameters:
        occupancy: int
            Number of people in the room

        proba_infectious: float or list[float]
            float: The probability that an individual is infectious
            list[float]: Probabilitiest that each individual is infectious


    """
    try:
        len(proba_infectious)

        return 1 - (1 - proba_infectious).prod()
    except TypeError:

        if occupancy is None:
            raise "proba_infectious is not a list. Please pass in occupancy."
        return 1 - (1 - proba_infectious) ** occupancy

def remaining_risk_budget(total_risk_budget, risks):
    """
    total_risk_budget = 1 - (NOT risk 1)(NOT risk 2) * ... * (NOT risk N)
    (1 - total_risk_budget) = [(NOT risk 1) * (NOT risk 2)...]

    (1 - total_risk_budget) = [(NOT risk 1) * (NOT risk 2)...(NOT risk N-1) * (NOT risk N)]
    (1 - total_risk_budget) / [(NOT risk 1) * (NOT risk 2)...(NOT risk N-1)] = (NOT risk N)
    1 - (NOT risk N) = remaining budget
    """
    remaining_budget = 1-(1 - total_risk_budget) / (1 - risks).prod()

    if remaining_budget < 0:
        raise UserError("Budget exceeded!")
    return remaining_budget

def steady_state_co2_level(ventilation_cadr, co2_generation_rate, ambient_co2_ppm=420):
    """
    Returns: steady state CO2 in parts per million

    Parameters:
        ventilation_cadr: int or float
            The clean air delivery rate (CADR) from ventilation, in m3 per hour.

        co2_generation_rate: float
            The amount of CO2 being generated, in m3 per hour.

        ambient_co2_ppm: int or float
            The CO2 levels outside.

            Defaults to 420
    """
    return (co2_generation_rate / ventilation_cadr) * 1000000  + ambient_co2_ppm

def ventilation_cadr_m3_h(steady_state_co2_ppm, co2_generation_rate, ambient_co2_ppm=420):
    """
    Returns the ventilation clean air delivery rate in terms of cubic meters per hour.

    Parameters:
        steady_state_co2_ppm: int
            The steady state CO2 level.

        co2_generation_rate: float
            The amount of CO2 being generated, in m3 per hour.

        ambient_co2_ppm: int or float
            The CO2 levels outside.

            Defaults to 420
    """
    return (1000000 *  co2_generation_rate) / (steady_state_co2_ppm - ambient_co2_ppm)
