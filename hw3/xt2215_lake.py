import random
import math
import sys
import copy


class FrozenLake(object):

    def __init__(self, width, height, start, targets, blocked, holes):
        self.initial_state = start
        self.width = width
        self.height = height
        self.targets = targets
        self.holes = holes
        self.blocked = blocked

        self.actions = ('n', 's', 'e', 'w')
        self.states = set()
        for x in range(width):
            for y in range(height):
                if (x, y) not in self.targets and (x, y) not in self.holes and (x, y) not in self.blocked:
                    self.states.add((x, y))

        # Parameters for the simulation
        self.gamma = 0.9
        self.success_prob = 0.8
        self.hole_reward = -5.0
        self.target_reward = 1.0
        self.living_reward = -0.1

    # Internal functions for running policies
    def get_transitions(self, state, action):
        """
        Return a list of (successor, probability) pairs that
        can result from taking action from state
        """
        result = []
        x, y = state
        remain_p = 0.0

        if action == "n":
            success = (x, y - 1)
            fail = [(x + 1, y), (x - 1, y)]
        elif action == "s":
            success = (x, y + 1)
            fail = [(x + 1, y), (x - 1, y)]
        elif action == "e":
            success = (x + 1, y)
            fail = [(x, y - 1), (x, y + 1)]
        elif action == "w":
            success = (x - 1, y)
            fail = [(x, y - 1), (x, y + 1)]

        if success[0] < 0 or success[0] > self.width - 1 or \
                success[1] < 0 or success[1] > self.height - 1 or \
                success in self.blocked:
            remain_p += self.success_prob
        else:
            result.append((success, self.success_prob))

        for i, j in fail:
            if i < 0 or i > self.width - 1 or \
                    j < 0 or j > self.height - 1 or \
                    (i, j) in self.blocked:
                remain_p += (1 - self.success_prob) / 2
            else:
                result.append(((i, j), (1 - self.success_prob) / 2))

        if remain_p > 0.0:
            result.append(((x, y), remain_p))
        return result

    def move(self, state, action):
        """
        Return the state that results from taking this action
        """
        transitions = self.get_transitions(state, action)
        new_state = random.choices([i[0] for i in transitions], weights=[i[1] for i in transitions])
        return new_state[0]

    def simple_policy_rollout(self, policy):
        """
        Return (Boolean indicating success of trial, total rewards) pair
        """
        state = self.initial_state
        rewards = 0
        while True:
            if state in self.targets:
                return (True, rewards + self.target_reward)
            if state in self.holes:
                return (False, rewards + self.hole_reward)
            state = self.move(state, policy[state])
            rewards += self.living_reward

    def QValue_to_value(self, Qvalues):
        """
        Given a dictionary of q-values corresponding to (state, action) pairs,
        return a dictionary of optimal values for each state
        """
        values = {}
        for state in self.states:
            values[state] = -float("inf")
            for action in self.actions:
                values[state] = max(values[state], Qvalues[(state, action)])
        return values

    # Some useful functions for you to visualize and test your MDP algorithms
    def test_policy(self, policy, t=500):
        """
        Following the policy t times, return (Rate of success, average total rewards)
        """
        numSuccess = 0.0
        totalRewards = 0.0
        for i in range(t):
            result = self.simple_policy_rollout(policy)
            if result[0]:
                numSuccess += 1
            totalRewards += result[1]
        return (numSuccess / t, totalRewards / t)

    def get_random_policy(self):
        """
        Generate a random policy.
        """
        policy = {}
        for i in range(self.width):
            for j in range(self.height):
                policy[(i, j)] = random.choice(self.actions)
        return policy

    def gen_rand_set(self, width, height, size):
        """
        Generate a random set of grid spaces.
        Useful for creating randomized maps.
        """
        mySet = set([])
        while len(mySet) < size:
            mySet.add((random.randint(0, width), random.randint(0, height)))
        return mySet

    def print_map(self, policy=None):
        """
        Print out a map of the frozen pond, where * indicates start state,
        T indicates target states, # indicates blocked states, and O indicates holes.
        A policy may optimally be provided, which will be printed out on the map as well.
        """
        sys.stdout.write(" ")
        for i in range(2 * self.width):
            sys.stdout.write("--")
        sys.stdout.write("\n")
        for j in range(self.height):
            sys.stdout.write("|")
            for i in range(self.width):
                if (i, j) in self.targets:
                    sys.stdout.write("T\t")
                elif (i, j) in self.holes:
                    sys.stdout.write("O\t")
                elif (i, j) in self.blocked:
                    sys.stdout.write("#\t")
                else:
                    if policy and (i, j) in policy:
                        a = policy[(i, j)]
                        if a == "n":
                            sys.stdout.write("^")
                        elif a == "s":
                            sys.stdout.write("v")
                        elif a == "e":
                            sys.stdout.write(">")
                        elif a == "w":
                            sys.stdout.write("<")
                        sys.stdout.write("\t")
                    elif (i, j) == self.initial_state:
                        sys.stdout.write("*\t")
                    else:
                        sys.stdout.write(".\t")
            sys.stdout.write("|")
            sys.stdout.write("\n")
        sys.stdout.write(" ")
        for i in range(2 * self.width):
            sys.stdout.write("--")
        sys.stdout.write("\n")

    def print_values(self, values):
        """
        Given a dictionary {state: value}, print out the values on a grid
        """
        for j in range(self.height):
            for i in range(self.width):
                if (i, j) in self.holes:
                    value = self.hole_reward
                elif (i, j) in self.targets:
                    value = self.target_reward
                elif (i, j) in self.blocked:
                    value = 0.0
                else:
                    value = values[(i, j)]
                print("%10.2f" % value, end='')
            print()

    # Your code starts here
    def value_iteration(self, threshold=0.001):
        """
        The value iteration algorithm to iteratively compute an optimal
        value function for all states.
        """
        values = dict((state, 0.0) for state in self.states)
        index = 0;
        while True:
            # backup last time values with values now
            last_values = copy.copy(values);
            for now_state in values:
                # go through all the possible actions to find max value
                max_value = -sys.maxsize;
                for action in self.actions:
                    successors = self.get_transitions(now_state, action);
                    # go through all the s', sum
                    sum_value = 0;
                    for (state, prob) in successors:
                        reward = self.living_reward;
                        # check terminal states
                        if state in self.holes:
                            sum_value += prob * (reward + self.gamma * self.hole_reward);
                        elif state in self.targets:
                            sum_value += prob * (reward + self.gamma * self.target_reward);
                        else:
                            sum_value += prob * (reward + self.gamma * last_values[state]);

                    if sum_value > max_value:
                        max_value = sum_value;

                # update values
                values[now_state] = max_value;

            # compare value change between value now and last values
            under_threshold = True;
            for now_state in values:
                last_prob = last_values[now_state];
                now_prob = values[now_state];
                if abs(now_prob - last_prob) > threshold:
                    under_threshold = False;
                    break;

            # if no state values change by more than threshold, stop iteration on values;
            if under_threshold:
                break;
            index += 1;
        return values

    def extract_policy(self, values):
        """
        Given state values, return the best policy.
        """
        policy = {}
        # go through states to find policy for each state
        for now_state in self.states:
            # go through all the actions and find the argmax on action
            max_value = -sys.maxsize;
            max_action = '';
            for action in self.actions:
                successors = self.get_transitions(now_state, action);
                sum_value = 0;
                for (state, prob) in successors:
                    reward = self.living_reward;
                    if state in self.holes:
                        sum_value += prob * (reward + self.gamma * self.hole_reward);
                    elif state in self.targets:
                        sum_value += prob * (reward + self.gamma * self.target_reward);
                    else:
                        sum_value += prob * (reward + self.gamma * values[state]);

                # store the argmax
                if sum_value > max_value:
                    max_value = sum_value;
                    max_action = action;

            policy[now_state] = max_action;
        return policy

    def Qlearner(self, alpha, epsilon, num_robots):
        """
        Implement Q-learning with the alpha and epsilon parameters provided.
        Runs number of episodes equal to num_robots.
        """
        Qvalues = {}
        for state in self.states:
            for action in self.actions:
                Qvalues[(state, action)] = 0

        # YOUR CODE HERE
        remain_robots = num_robots;
        start_state = (0, 0);
        reward = self.living_reward;
        step = 0.1
        while remain_robots > 0:
            # every robot starts from state(0,0);
            state = start_state;
            while True:
                if state in self.holes or state in self.targets:
                    break;

                # select action
                random_epsilon = random.random();
                if random_epsilon < epsilon:
                    # select a random action
                    random_action = random.choice(self.actions);
                    action = random_action;
                else:
                    # select the best action
                    max_q_value = -sys.maxsize;
                    best_action = '';
                    for a in self.actions:
                        q_value = Qvalues[(state, a)];
                        if q_value > max_q_value:
                            max_q_value = q_value;
                            best_action = a;
                    action = best_action;
                next_state = self.move(state, action);
                # sample (state, action, next_state, reward)
                if next_state in holes:
                    Qvalues[(state, action)] = (1 - alpha) * Qvalues[(state, action)] + alpha * (
                            reward + self.gamma * self.hole_reward);
                elif next_state in targets:
                    Qvalues[(state, action)] = (1 - alpha) * Qvalues[(state, action)] + alpha * (
                            reward + self.gamma * self.target_reward);
                else:
                    max_q_value = -sys.maxsize;
                    for a in self.actions:
                        q_value = Qvalues[(next_state, a)];
                        if q_value > max_q_value:
                            max_q_value = q_value;
                    Qvalues[(state, action)] = (1 - alpha) * Qvalues[(state, action)] + alpha * (
                            reward + self.gamma * max_q_value);

                state = next_state;

            alpha -= alpha*step;
            epsilon -= epsilon*step;
            remain_robots -= 1;
        return Qvalues


if __name__ == "__main__":
    # Create a lake simulation
    width = 8
    height = 8
    start = (0, 0)
    targets = set([(3, 4)])
    blocked = set([(3, 3), (2, 3), (2, 4)])
    holes = set([(4, 0), (4, 1), (3, 0), (3, 1), (6, 4), (6, 5), (0, 7), (0, 6), (1, 7)])
    lake = FrozenLake(width, height, start, targets, blocked, holes)

    rand_policy = lake.get_random_policy()
    lake.print_map()
    lake.print_map(rand_policy)
    print(lake.test_policy(rand_policy))

    opt_values = lake.value_iteration()
    lake.print_values(opt_values)
    opt_policy = lake.extract_policy(opt_values)
    lake.print_map(opt_policy)
    print(lake.test_policy(opt_policy))
    '''
     when num_robots = 100, it seldom returns a total reward less than -1 but still does;
    '''
    # Qvalues = lake.Qlearner(alpha=0.5, epsilon=0.5, num_robots=100)
    # learned_values = lake.QValue_to_value(Qvalues)
    # learned_policy = lake.extract_policy(learned_values)
    # lake.print_map(learned_policy)
    # print(lake.test_policy(learned_policy))

    '''
     decreasing alpha and epsilon
     at first,epsilon equals to 1, which means the robot takes completely random moves to explore the state space maximally
     
    '''
    Qvalues = lake.Qlearner(alpha=0.8, epsilon=0.8, num_robots=30)
    learned_values = lake.QValue_to_value(Qvalues)
    learned_policy = lake.extract_policy(learned_values)
    lake.print_map(learned_policy)
    print(lake.test_policy(learned_policy))
