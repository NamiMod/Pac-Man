# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

# Edited by : Seyed Nami Modarressi

from os import stat
import mdp, util

from learningAgents import ValueEstimationAgent
import collections

MIN_NUMBER = -9999
MAX_NUMBER = 9999

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        for i in range(0,self.iterations):
            counter = util.Counter()
            for s in self.mdp.getStates():
                max = MIN_NUMBER
                for a in self.mdp.getPossibleActions(s):
                    q = self.computeQValueFromValues(s, a)
                    if max < q :
                        counter[s] = q
                        max = q
                    else :
                        counter[s] = max
            self.values = counter

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        q = 0
        for node in self.mdp.getTransitionStatesAndProbs(state, action):
            r = self.mdp.getReward(state, action, node[0])
            next_node_q_value = self.values[node[0]]
            q = q + node[1] * (r + next_node_q_value * self.discount)
        return q

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        max = MIN_NUMBER
        action = None
        for a in self.mdp.getPossibleActions(state):
            q = self.computeQValueFromValues(state, a)
            if max < q :
                action = a
                max = q
        if self.mdp.isTerminal(state):
            action = None
            self.values[state] = 0
        return action

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        states = self.mdp.getStates()
        for i in range(0,self.iterations):
            index = i % len(states)
            if self.mdp.isTerminal(states[index]) == True:
                continue
            else :
                actions = self.mdp.getPossibleActions(states[index])
                max = MIN_NUMBER
                for j in actions :
                    temp = self.getQValue(states[index],j)
                    if max < temp:
                        max = temp
                self.values[states[index]] = max

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        queue = util.PriorityQueue()
        p_states = {} 
        for s in self.mdp.getStates():
            if self.mdp.isTerminal(s) == True:
                continue
            else:
                actions = self.mdp.getPossibleActions(s)
                for a in actions :
                    for node in self.mdp.getTransitionStatesAndProbs(s, a):
                        flag = False
                        for temp in p_states:
                            if temp == node[0]:
                                flag = True
                                break
                        if flag == True :
                            p_states[node[0]].add(s)
                        else:
                            p_states[node[0]] = {s}

        for s in self.mdp.getStates():
            if self.mdp.isTerminal(s) == True :
                continue
            else :
                value = self.values[s]
                temp = []
                for action in self.mdp.getPossibleActions(s):
                    temp.append(self.computeQValueFromValues(s, action))
                max_q = max(temp)
                diff = abs(value - max_q)
                queue.update(s, -diff)

        for i in range(0, self.iterations):
            if queue.isEmpty():
                break
            else :
                state = queue.pop()
                if self.mdp.isTerminal(state) == True:
                    continue
                else:
                    temp = []
                    for a in self.mdp.getPossibleActions(state) :
                        temp.append(self.computeQValueFromValues(state, a))
                    self.values[state] = max (temp)

                for s in p_states[state]:
                    if self.mdp.isTerminal(s) == True:
                        continue
                    else :
                        v = self.values[s]
                        t = []
                        for a in self.mdp.getPossibleActions(s) :
                            t.append(self.computeQValueFromValues(s, a))
                        m = max(t)
                        diff = abs(v - m)
                        if diff > self.theta:
                            queue.update(s, -diff)