classdef PriorityQueue < handle
    properties (Access = private)
        elements
        priorities
        size
    end
    
    methods
        function obj = PriorityQueue()
            obj.elements = {};
            obj.priorities = [];
            obj.size = 0;
        end
        
        function insert(obj, element, priority)
            % Insert element with priority
            obj.size = obj.size + 1;
            obj.elements{obj.size} = element;
            obj.priorities(obj.size) = priority;
        end
        
        function element = pop(obj)
            % Pop element with lowest priority
            if obj.isEmpty()
                element = [];
                return;
            end
            
            [~, idx] = min(obj.priorities);
            element = obj.elements{idx};
            
            % Remove element
            obj.elements(idx) = [];
            obj.priorities(idx) = [];
            obj.size = obj.size - 1;
        end
        
        function result = isEmpty(obj)
            % Check if queue is empty
            result = (obj.size == 0);
        end
    end
end