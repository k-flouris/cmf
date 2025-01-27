For testing metric:
    
        
        place desired runs in some folder e.g. runs/metric_test/
        run from driver-src driver-test-metric.py  (or metric evaluate if it works
            ->it will produce some results - change visualizer.py accoridgly
        run from analysis run collect-results-metric.py with disired options
        
For ood:
    
        run from driver-src driver-test-complete.py
            -> should have --test-ood as option
        run from analysis run collect-results-ood.py with disired options

For fid:
        
        similar to above
        
NB: for tabular data the tabular-evaluate.py evaluates on cpu without need to --test-fid first.
        
For training:
    
        run driver-src/driver* train 

see also useful-commands for other commands
