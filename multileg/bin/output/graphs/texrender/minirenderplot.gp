set yrange [0:2]
set style line 11 lc rgb '#808080' lt 1
set border 3 back ls 11
set tics nomirror
# define grid
set style line 12 lc rgb '#808080' lt 0 lw 1
set grid back ls 12

set style line 1 lc rgb '#8b9946' pt -1 ps 1 lt 1 lw 3 # --- green
set style line 2 lc rgb '#5c1720' pt -1 ps 1 lt 1 lw 3 # --- red
set style line 3 lc rgb '#76a1c8' pt -1 ps 1 lt 1 lw 3 # --- blue
set style line 4 lc rgb '#FF5B00' pt -1 ps 1 lt 1 lw 3 # --- orange (for fit)

set style line 11 lc rgb '#8b9946' pt -1 ps 1 lt 1 lw 0.3 # --- red
set style line 22 lc rgb '#5c1720' pt -1 ps 1 lt 1 lw 0.3 # --- green
set style line 33 lc rgb '#76a1c8' pt -1 ps 1 lt 1 lw 0.3 # --- blue

set style fill transparent solid 0.2 border border 5#5lt6#6
# noborder
set key top left
set xlabel 'Step'
set ylabel 'Milliseconds'

set ytics 0.5
# font "Verdana,12" 
#set xtics
# font "Verdana,12" 
plot \
"../perf_serial.gnuplot.txt" using 1:4:5 with filledcurves title 'serial error', "" using 1:2 with lines ls 1 t 'Serial5', \
"../perf_parallel.gnuplot.txt" using 1:4:5 with filledcurves title 'parallel error', "" using 1:2 with lines ls 2 t 'Parallel5'