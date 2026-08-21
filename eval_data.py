"""Held-out reflection buffer + queries for LASR evaluation.

Each of the 21 MetaBuffer-Math problem types has:
  - 3 buffer reflections (worked example + strategy), none of which
    reuse the query wording
  - 3 held-out agent queries (new problem statements)

Gold for a query is any reflection of the same problem type. Distractors
are off-type reflections so ANN must actually discriminate.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

from encode_templates import get_templates_from_file

TYPE_NAMES = {
    1: "Normalization",
    2: "Total Quantity",
    3: "Sum and Difference",
    4: "Sum and Multiple",
    5: "Difference and Multiple",
    6: "Proportional",
    7: "Meeting",
    8: "Overtaking",
    9: "Tree Planting",
    10: "Age",
    11: "Boat",
    12: "Train Passing a Bridge",
    13: "Clock",
    14: "Surplus and Deficit",
    15: "Work",
    16: "Cows Eating Grass",
    17: "Chicken and Rabbit",
    18: "Profit and Loss",
    19: "Bank Interest",
    20: "Solution Concentration",
    21: "Equation",
}

# Extra worked reflections (original example is added from example_temp.txt).
EXTRA_REFLECTIONS: Dict[int, List[str]] = {
    1: [
        "Problem: 8 bottles of milk cost 36 yuan. How much do 15 bottles cost?\n"
        "Reflection: Normalization / unit-rate. Find the cost of one bottle (36/8=4.5), then scale to 15 (4.5*15=67.5). Do not start from a guessed total.",
        "Problem: A printer finishes 240 pages in 6 minutes. How many pages in 25 minutes at the same rate?\n"
        "Reflection: First reduce to a single-minute rate (240/6=40 pages/min), then multiply by the requested time. Unit first, then scale.",
    ],
    2: [
        "Problem: After switching from 4.5 L to 3.6 L per tank, how many tanks can be filled with the fuel that used to fill 80 tanks?\n"
        "Reflection: Total-quantity. Compute the conserved total (4.5*80=360 L), then divide by the new per-unit use (360/3.6=100).",
        "Problem: A farm harvested 2.4 tonnes per acre on 65 acres. After a new method yields 3.0 tonnes per acre, how many acres produce the same total harvest?\n"
        "Reflection: Hold the total harvest fixed (2.4*65), then divide by the new per-acre yield. The hidden total is the bridge.",
    ],
    3: [
        "Problem: Two boxes hold 150 kg together. Box A is 18 kg heavier than box B. What is each weight?\n"
        "Reflection: Sum-and-difference. Larger=(150+18)/2=84, smaller=(150-18)/2=66. Recover parts from total and gap.",
        "Problem: The sum of two readings is 540 and they differ by 90. Find both readings.\n"
        "Reflection: Half the sum-plus-difference is the larger value; half the sum-minus-difference is the smaller. Do not treat it as a multiple.",
    ],
    4: [
        "Problem: 360 seats are split so that class A has 5 times as many seats as class B. How many does each class have?\n"
        "Reflection: Sum-and-multiple. Smaller=360/(5+1)=60, larger=60*5=300. Partition a known total into (m+1) shares.",
        "Problem: A 840-page report is such that volume 1 has four times the pages of volume 2. Find each volume.\n"
        "Reflection: Let the smaller share be the total divided by (multiple+1). The known quantity is the SUM, not the difference.",
    ],
    5: [
        "Problem: Red marbles are 4 times the blue ones, and there are 90 more red than blue. How many of each?\n"
        "Reflection: Difference-and-multiple. Smaller=90/(4-1)=30, larger=120. The known quantity is the GAP, not the total.",
        "Problem: A truck carries 7 times the load of a cart and 540 kg more. Find both loads.\n"
        "Reflection: Divide the difference by (multiple-1) to recover the smaller amount, then scale. Do not divide a missing total.",
    ],
    6: [
        "Problem: 25 kg of wheat make 18 kg of flour. How much flour from 400 kg of wheat?\n"
        "Reflection: Proportional. Scale factor=400/25=16, flour=18*16=288. Same-type quantities, find the multiple, then map.",
        "Problem: 12 machines produce 90 parts. How many parts do 28 machines produce at the same rate?\n"
        "Reflection: Compute how many times 28 is of 12, then multiply that factor through the known output. Keep the ratio intact.",
    ],
    7: [
        "Problem: Towns are 270 km apart. Cars leave both towns at once at 50 km/h and 40 km/h toward each other. When do they meet?\n"
        "Reflection: Meeting. Time=distance/(v1+v2)=270/90=3 h. Opposite directions, closing speed is the SUM of speeds.",
        "Problem: Two swimmers 1.2 km apart swim toward each other at 0.7 km/h and 0.5 km/h. Meeting time?\n"
        "Reflection: Add the speeds because they eat the gap from both ends. Meeting is not an overtaking (same-direction) setup.",
    ],
    8: [
        "Problem: A courier at 90 km/h chases a truck at 60 km/h that left 2 hours earlier. Catch-up time?\n"
        "Reflection: Overtaking. Head-start distance=60*2=120 km, relative speed=90-60=30, time=4 h. Same direction: subtract speeds.",
        "Problem: A fast train 80 km/h pursues a slow train 55 km/h that is already 150 km ahead. Hours to catch?\n"
        "Reflection: Overtaking distance over speed difference. Do not add the speeds; they are not closing head-on.",
    ],
    9: [
        "Problem: A 240 m fence gets a post every 6 m, including both ends. How many posts?\n"
        "Reflection: Linear tree-planting. Count=distance/spacing+1=41. Endpoints add the extra post; this is not a circular count.",
        "Problem: A circular fountain rim is 90 m; lamps every 5 m. How many lamps?\n"
        "Reflection: Circular planting. Count=distance/spacing=18, no extra +1 because start and end coincide.",
    ],
    10: [
        "Problem: Maya is 7 years older than Leo. Maya is 31. How old was Leo when Maya was 20?\n"
        "Reflection: Age. The gap is invariant. Leo is 24 now; when Maya was 20, Leo was 13. Convert via the constant difference.",
        "Problem: A father is 28 years older than his son. In 8 years the father will be 50. Son's current age?\n"
        "Reflection: Future father=50 so now=42; son=42-28=14. Age-difference problems are not sum-and-multiple unless a ratio is given.",
    ],
    11: [
        "Problem: Downstream 180 km takes 6 h; current is 5 km/h. Time for the same distance upstream?\n"
        "Reflection: Boat. Still-water speed=180/6-5=25, upstream=20, time=9 h. Downstream adds current; upstream subtracts it.",
        "Problem: A boat's still-water speed is 16 km/h, current 4 km/h. How long is 48 km upstream?\n"
        "Reflection: Upstream speed=16-4=12, time=4 h. Keep boat speed and current speed as separate additives, not a meeting sum.",
    ],
    12: [
        "Problem: A 1.8 km tunnel is crossed by a 200 m train at 20 m/s. How many seconds to fully pass?\n"
        "Reflection: Train-and-bridge. Distance=train+tunnel=2000 m, time=2000/20=100 s. The train's own length must be added.",
        "Problem: A 250 m train at 90 km/h completely passes a 650 m platform. Time in seconds?\n"
        "Reflection: Convert 90 km/h=25 m/s. Time=(250+650)/25=36 s. Passing means covering train length plus obstacle length.",
    ],
    13: [
        "Problem: After 3 o'clock, when do the hour and minute hands first overlap?\n"
        "Reflection: Clock. At 3:00 the hour hand leads by 15 divisions. Catch-up time=15/(1-1/12)=16.36 min. Minute hand gains 11/12 per minute.",
        "Problem: Starting at 6 o'clock, after how many minutes are the hands in a straight line (180°)?\n"
        "Reflection: Convert to a catch-up/difference problem on the clock face. Relative speed is 11/12 divisions per minute; 6:00 already has a 180° offset so solve the residual gap.",
    ],
    14: [
        "Problem: 5 sweets each leaves 8 extra; 6 each is 4 short. How many children and sweets?\n"
        "Reflection: Surplus-and-deficit. Children=(8+4)/(6-5)=12, sweets=5*12+8=68. Combine leftover and shortage over the per-person change.",
        "Problem: 9 books each leaves 7 spare; 11 each falls 5 short. Find the class size.\n"
        "Reflection: (surplus+deficit)/difference in share= (7+5)/(11-9)=6. This is not a chicken-rabbit heads-and-legs count.",
    ],
    15: [
        "Problem: A does a job in 12 days, B in 18. Days together?\n"
        "Reflection: Work. Treat the job as 1. Rate=1/12+1/18=5/36, time=36/5=7.2 days. Invert the summed rates.",
        "Problem: Pipe A fills a tank in 8 h, pipe B in 24 h. Hours to fill together?\n"
        "Reflection: Combined rate=1/8+1/24=1/6, time=6 h. Work problems add efficiencies, not distances.",
    ],
    16: [
        "Problem: 12 sheep finish a pasture in 15 days, 18 sheep in 8 days. How many sheep finish it in 6 days if grass keeps growing?\n"
        "Reflection: Growing-grass / Newton. Daily growth=(12*15-18*8)/(15-8)=12/7. Recover initial stock, add 6-day growth, divide by 6.",
        "Problem: 8 cows last 20 days, 12 cows last 10 days on a growing field. Cows that last 8 days?\n"
        "Reflection: The resource is not a fixed work-1. Split standing grass from daily growth using the two observations, then reallocate.",
    ],
    17: [
        "Problem: 40 heads and 100 legs. How many chickens and rabbits?\n"
        "Reflection: Chicken-and-rabbit. If all chickens, 20 extra legs => 10 rabbits, 30 chickens. Two species, heads + legs, not surplus/deficit.",
        "Problem: 50 animals, 140 legs, chickens and goats (4 legs). How many goats?\n"
        "Reflection: Extra legs beyond 2 each are 40, so goats=20. Classic heads-and-legs elimination, not a sum-and-difference of two counts only.",
    ],
    18: [
        "Problem: Cost 80 yuan, sold at 20% profit. Later marked 15% off that selling price. Final vs cost?\n"
        "Reflection: Profit-and-loss. Sell1=80*1.2=96, sell2=96*0.85=81.6, net +2%. Chain percentage changes on the price path, not on a solvent.",
        "Problem: A dealer buys at 250 and wants 12% profit after a 10% discount on the marked price. What marked price?\n"
        "Reflection: Target sell=250*1.12=280=0.9*marked, marked=311.11. Profit is (sell-cost)/cost, not an interest-time product.",
    ],
    19: [
        "Problem: 5000 yuan at 0.6% monthly. After the term the withdrawal is 5720. How many months?\n"
        "Reflection: Bank interest. Interest=720, rate-sum=720/5000=0.144, months=0.144/0.006=24. Interest=principal*rate*time.",
        "Problem: Principal 2000 yuan grows to 2240 in 10 months. Monthly rate?\n"
        "Reflection: Interest=240, monthly rate=240/2000/10=1.2%. Invert the linear interest identity; this is not a profit-margin on cost.",
    ],
    20: [
        "Problem: 80 g of 20% brine. How much water to reach 8%?\n"
        "Reflection: Concentration. Solute is invariant (16 g). New solution=16/0.08=200 g, water added=120 g. Dilute by adding solvent, not by a profit markup.",
        "Problem: 120 g of 15% acid. How many grams of 15% must be evaporated (water only) to reach 25%?\n"
        "Reflection: Solute stays 18 g. New mass=18/0.25=72 g, so 48 g water leaves. Track solute vs solution, not principal vs interest.",
    ],
    21: [
        "Problem: Two numbers sum to 75. The first is 15 less than twice the second. Find them.\n"
        "Reflection: Equation. Let second=x, first=75-x=2x-15 => x=30, first=45. Set a variable, write equality, solve, check.",
        "Problem: A is 12 more than B and 3 times B is 18 more than A. Find A and B.\n"
        "Reflection: Translate both sentences into equations (A=B+12, 3B=A+18) and solve the system. The method is 'set-formulate-solve', not a canned sum-difference formula alone.",
    ],
}

QUERIES: Dict[int, List[str]] = {
    1: [
        "Nine identical notebooks sell for 27 rupees. What is the price of 14 such notebooks?",
        "A tap fills 45 litres in 9 minutes. How many litres in 16 minutes at that rate?",
        "Buying 7 mangoes costs 56. How much for 11 mangoes of the same kind?",
    ],
    2: [
        "A mill used 6.5 kg of grain per sack and packed 120 sacks. The new pack uses 5 kg per sack. How many new sacks from the same grain?",
        "Wire that used to make 45 pieces of 1.2 m now makes pieces of 0.9 m. How many new pieces?",
        "Paint that covered 36 boards at 0.8 L each is now applied at 0.6 L each. How many boards can be covered?",
    ],
    3: [
        "Two tanks hold 260 litres together and the first has 40 litres more than the second. Find both volumes.",
        "The sum of two scores is 188 and the difference is 24. What are the scores?",
        "A and B together weigh 142 kg; A is 16 kg heavier. Find A and B.",
    ],
    4: [
        "320 trees are planted so that pines are three times the oaks. How many of each?",
        "A 540-dollar bill is split so that one share is eight times the other. Find the shares.",
        "In a 210-student cohort, seniors are twice the juniors. How many seniors and juniors?",
    ],
    5: [
        "Apples are five times the oranges, and there are 48 more apples than oranges. How many of each?",
        "A large gear has 6 times the teeth of a small gear and 75 more teeth. Find both counts.",
        "Town X has 4 times the buses of town Y and 96 extra buses. How many buses in each town?",
    ],
    6: [
        "15 kg of seed yield 9 kg of oil. How much oil from 250 kg of seed?",
        "8 identical pumps move 120 cubic metres. How much do 22 pumps move?",
        "35 kg of ore give 14 kg of metal. Metal from 210 kg of the same ore?",
    ],
    7: [
        "Two cyclists 84 km apart ride toward each other at 18 km/h and 24 km/h. Hours until they meet?",
        "Trains leave two stations 450 km apart at the same time, 70 km/h and 80 km/h, heading toward each other. Meeting time?",
        "Two boats 36 km apart steam toward each other at 11 km/h and 13 km/h. When do they meet?",
    ],
    8: [
        "A car at 100 km/h starts 3 hours after a bus at 55 km/h on the same road. When does the car catch the bus?",
        "Runner A at 6 m/s chases runner B at 4.5 m/s who is 90 m ahead. Catch-up time?",
        "A police bike at 120 km/h pursues a car at 90 km/h that has a 45 km head start. Hours to overtake?",
    ],
    9: [
        "A 180 m path has trees every 4 m with trees at both ends. How many trees?",
        "A 64 m wall is lined with hooks every 8 m, including both corners. How many hooks?",
        "Around a 120 m circular pond, flags stand every 6 m. How many flags?",
    ],
    10: [
        "Nora is 9 years older than Priya. Nora is 27. How old was Priya when Nora was 18?",
        "A mother is 26 years older than her child. In 6 years the mother will be 40. Child's age now?",
        "Sam is 4 years younger than Riya. Riya is 19. How old will Sam be when Riya is 30?",
    ],
    11: [
        "A steamer goes 210 km downstream in 7 hours. The current is 6 km/h. Hours for 210 km upstream?",
        "Still-water speed 22 km/h, current 3 km/h. Time to cover 76 km upstream?",
        "Downstream 14 km/h, upstream 8 km/h. What is the current speed?",
    ],
    12: [
        "How many seconds does a 180 m train at 54 km/h take to pass a 720 m bridge completely?",
        "A 300 m train at 15 m/s goes through a 600 m tunnel. Time to clear the tunnel?",
        "A train 120 m long at 72 km/h passes a standing person. How many seconds does it take?",
    ],
    13: [
        "Starting from 2 o'clock, after how many minutes do the hour and minute hands overlap?",
        "After 5 o'clock, when are the hands first at right angles?",
        "From 10 o'clock, after how many minutes do the hands overlap?",
    ],
    14: [
        "Giving 8 pencils each leaves 5 spare; giving 9 each is 3 short. How many pupils and pencils?",
        "If 4 cakes each remain 6, if 5 each fall 4 short. Find the number of children.",
        "7 notebooks each leave 9 extra; 10 each fall 6 short. How many students?",
    ],
    15: [
        "A finishes in 20 days, B in 30 days. How many days working together?",
        "One tap fills in 10 hours, another in 15 hours. Hours to fill the tank together?",
        "A can mow a lawn in 6 hours, B in 3 hours. Time if they mow together?",
    ],
    16: [
        "9 cows graze a field in 16 days, 12 cows in 10 days. How many cows finish it in 8 days if grass grows daily?",
        "A meadow lasts 7 oxen 24 days or 14 oxen 10 days. Oxen that last 12 days?",
        "15 horses eat a growing pasture in 12 days, 20 horses in 8 days. Horses for 6 days?",
    ],
    17: [
        "There are 28 heads and 80 legs among chickens and rabbits. How many rabbits?",
        "36 animals, 100 legs, birds (2 legs) and beasts (4 legs). How many beasts?",
        "A pen has chickens and sheep, 23 heads and 62 legs. How many sheep?",
    ],
    18: [
        "An item bought for 400 is sold at 25% profit, then a 20% discount is given on that selling price. Net percent vs cost?",
        "Cost is 180. What selling price gives a 15% profit?",
        "Marked price 500, sold at 12% off. If cost was 400, what is the profit percent?",
    ],
    19: [
        "1800 yuan at 0.5% per month becomes 1980 at withdrawal. How many months was it deposited?",
        "Principal 2500 earns 300 interest in 8 months. Monthly interest rate?",
        "How much interest does 6000 yuan earn in 18 months at 0.7% per month?",
    ],
    20: [
        "60 grams of a 12% sugar solution. How much water to dilute it to 8%?",
        "40 g of 25% saline. How many grams of water should be added to get 10%?",
        "90 g of 30% acid. How much water evaporates to make it 45%?",
    ],
    21: [
        "Two classes total 70 students. Class A has 10 fewer than twice class B. Find each class.",
        "The larger of two numbers is 7 more than the smaller and their sum is 51. Write an equation and solve.",
        "Let x be the smaller integer. Three times x plus 12 equals 5 times (x-2). Find x.",
    ],
}

# Sibling-type hard negatives: same surface language, wrong identity.
HARD_NEGATIVES = [
    "Problem: Two numbers add to 240 and one is 4 times the other. How many of each?\n"
    "Reflection: Wrong template if you reach for sum-and-difference. The gap is not given; the SUM and a MULTIPLE are. Smaller=240/5.",
    "Problem: Red is 3 times blue and they differ by 80. Find both.\n"
    "Reflection: This is difference-and-multiple, not sum-and-difference. Dividing 80 by 2 treats a gap as a total and fails.",
    "Problem: Together 180 seats, A has 20 more than B. Find A and B.\n"
    "Reflection: Sum-and-difference, not chicken-and-rabbit. There are no heads/legs; recover parts from total and gap.",
    "Problem: Two cars 200 km apart, one 20 km/h faster, same direction. Catch-up time?\n"
    "Reflection: Overtaking, not meeting. Same direction means subtract speeds. Adding them is the meeting-formula trap.",
    "Problem: Two boats start at one dock and steam opposite ways at 12 and 16 km/h. When are they 84 km apart?\n"
    "Reflection: Opposite departing is a meeting-style SUM of speeds, not a current-speed boat problem. Time=84/28.",
    "Problem: A train 200 m long at 20 m/s meets an oncoming 180 m train at 10 m/s. Time to pass?\n"
    "Reflection: Closing speed is the SUM and lengths add, but this is a passing-trains variant, not a single train-plus-bridge template.",
    "Problem: Cost 200, marked up 25%, then 25% off. Net vs cost?\n"
    "Reflection: Profit-and-loss chaining, not bank interest. There is no principal*rate*time; the two percentages apply in sequence.",
    "Problem: 80 g of 10% acid plus 40 g water. New percent?\n"
    "Reflection: Concentration, solute invariant. Treating 10% as a profit markup or as monthly interest is the wrong identity.",
    "Problem: A and B finish a job in 8 and 24 days. Time together?\n"
    "Reflection: Work-rate, job=1. This is not Newton grass; the work does not grow while they labour.",
    "Problem: 10 cows last 20 days, 15 cows last 10 days on a growing field. Cows for 5 days?\n"
    "Reflection: Growing-grass. A plain work-together harmonic mean ignores daily growth and under-counts the herd.",
]

DISTRACTORS = HARD_NEGATIVES + [
    "Reflection: For a geometry proof, draw the figure, mark equal angles, and apply AA similarity before chasing lengths.",
    "Reflection: Counting subsets uses nCk. Check whether order matters; if it does, switch to permutations.",
    "Reflection: A probability tree splits on sequential events. Multiply along branches, add disjoint leaves.",
    "Reflection: Complete the square to rewrite a quadratic, then read the vertex and the minimum value.",
    "Reflection: In DFS vs BFS, choose BFS for shortest unweighted paths and DFS for cycle detection in a graph.",
    "Reflection: Modular arithmetic: reduce bases before multiplying. Use Fermat when the modulus is prime.",
    "Reflection: Similar triangles share angles; corresponding sides are proportional. Set the cross-multiply equation on the matching pair.",
    "Reflection: Inclusion-exclusion for two sets is |A|+|B|-|A∩B|. Do not add the intersection twice.",
    "Reflection: Convert a repeating decimal to a fraction by shifting the repeat window and subtracting.",
    "Reflection: The discriminant b^2-4ac tells the number of real roots. Do not confuse it with a profit margin.",
    "Reflection: For a 3D box diagonal use sqrt(l^2+w^2+h^2). This is not a train-length plus bridge-length sum.",
    "Reflection: Bayes updates P(H|E)=P(E|H)P(H)/P(E). Keep prior, likelihood, and evidence distinct.",
    "Reflection: Linear interpolation between two tabulated points is a weighted average of the endpoints.",
    "Reflection: A geometric series sums to a(1-r^n)/(1-r). Identify first term and common ratio first.",
    "Reflection: Constraint optimization with AM-GM: equality when terms are equal. Not a work-rate split.",
    "Reflection: In kinematics, s=ut+1/2at^2 for constant acceleration. This is not a meeting-time sum of speeds.",
    "Reflection: Hash collisions are handled by chaining or open addressing; load factor drives expected probes.",
    "Reflection: Binary search needs a monotonic predicate. Midpoint bias must be written to avoid infinite loops.",
    "Reflection: Matrix multiply is n^3 naive; use associativity to parenthesize a chain for the cheapest order.",
    "Reflection: For circle theorems, an angle in a semicircle is 90°. Mark the diameter before assigning right angles.",
    "Reflection: Log laws: log(ab)=log a+log b. Change-of-base when the bases differ. Not a concentration ratio.",
    "Reflection: Expected value is the probability-weighted sum of outcomes. Linearity holds even when dependent.",
    "Reflection: A recurrence a_n=2a_{n-1}+1 unrolls to a closed 2^n form. Solve by homogeneous plus particular.",
    "Reflection: Dimensional analysis: check SI units before trusting an equation. Cancel kg, m, s separately.",
    "Reflection: For integer overflow, widen the type or take mods at each step. Silent wrap is not a surplus/deficit.",
    "Reflection: The Chinese Remainder Theorem splits a system of congruences when moduli are coprime.",
    "Reflection: Convex hull of a point set can be Graham-scanned in n log n. Not a tree-planting endpoint count.",
    "Reflection: In optics, 1/f=1/v-1/u (sign convention). Do not treat this as a work-together harmonic sum blindly.",
    "Reflection: A Markov chain stationary distribution solves πP=π. This is not an age-gap invariance.",
    "Reflection: FFT multiplies polynomials in n log n. Evaluate at roots of unity, pointwise multiply, invert.",
    "Reflection: Kruskal grows an MST by sorted edges while union-find blocks cycles. Greedy stays optimal here.",
    "Reflection: For base conversion, repeated remainder by the new base; read digits in reverse.",
    "Reflection: Simpson's rule approximates definite integrals with parabolic panels. Error shrinks with h^4.",
    "Reflection: A generating function encodes a sequence as coefficients. Extract [x^n] after algebraic simplification.",
    "Reflection: SVD of a data matrix yields principal axes. That is a modelling tool, not a word-problem template.",
    "Reflection: Cache-aware blocking raises arithmetic intensity. This is hardware, not a rate-time-distance story.",
    "Reflection: Regular expressions cannot count arbitrary nesting; use a CFG/PDA when you need a stack.",
    "Reflection: Lagrange interpolation fits n+1 points with a degree-n polynomial. Watch Runge's phenomenon.",
    "Reflection: In special relativity, time dilation is γΔτ. Do not add frame speeds like a meeting problem.",
    "Reflection: A disjoint-set with union-by-rank and path compression is inverse-Ackermann amortized.",
]


def _extract_example(template: str) -> Tuple[str, str]:
    example = ""
    m = re.search(r"\*\*Example\*\*:\s*(.+?)(?:\n\n\*\*Solution\*\*|\n\n### Solution|\Z)", template, re.S)
    if m:
        example = re.sub(r"\s+", " ", m.group(1)).strip()
    strategy = ""
    sm = re.search(r"\*\*Solution Strategy\*\*:\s*(.+)", template)
    if sm:
        strategy = sm.group(1).strip()
    else:
        dm = re.search(r"\*\*Definition\*\*:\s*(.+)", template)
        strategy = dm.group(1).strip() if dm else "Apply the type's quantitative relations."
    return example, strategy


def load_eval_corpus(template_path: str = "data/example_temp.txt"):
    templates, labels = get_templates_from_file(template_path)
    reflections: List[Dict] = []
    rid = 0
    used = set()
    for tmpl, lab in zip(templates, labels):
        example, strategy = _extract_example(tmpl)
        name = TYPE_NAMES.get(lab, f"Type {lab}")
        text = (
            f"Problem: {example}\n"
            f"Reflection: This is a {name} problem. {strategy} "
            f"Reuse the type's identities rather than a generic equation hunt."
        )
        reflections.append({"id": f"orig-{lab}", "type": lab, "name": name, "text": text, "source": "original"})
        used.add(lab)
        rid += 1

    for lab, texts in EXTRA_REFLECTIONS.items():
        name = TYPE_NAMES[lab]
        for j, text in enumerate(texts, start=1):
            reflections.append(
                {"id": f"extra-{lab}-{j}", "type": lab, "name": name, "text": text, "source": "extra"}
            )

    queries: List[Dict] = []
    for lab, qs in QUERIES.items():
        name = TYPE_NAMES[lab]
        for j, text in enumerate(qs, start=1):
            queries.append({"id": f"q-{lab}-{j}", "type": lab, "name": name, "text": text})

    distractors: List[Dict] = []
    for j, text in enumerate(DISTRACTORS, start=1):
        distractors.append({"id": f"d-{j}", "type": 0, "name": "Distractor", "text": text, "source": "distractor"})

    corpus = reflections + distractors
    return reflections, queries, distractors, corpus
