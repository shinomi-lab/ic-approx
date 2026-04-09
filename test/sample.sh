function exec {
    gp=$1
    dir=$2
    max_ns=$3
    max_ps=$4
    iter=$5
    seed_num=$6
    ubp=$7
    name=$8
    echo $gp
    for ns in $(seq 1 $max_ns); do
        for ps in $(seq 1 $max_ps); do
            echo $name $ns $ps
            ./target/release/ic-approx \
                -g $gp -d $dir -u $ubp -n $seed_num --dsteps 10 --rpt 8\
                --ns $ns --ps $ps\
                --tot ./test/sample-result/${name}-${ns}-${ps}_time.csv\
                --toe ./test/sample-result/${name}-${ns}-${ps}_error.csv\
                mcm --iter $iter --mr 0 tyl --fn tyl0 --fn dmp --fn sssn --fn > ./test/sample-result/${name}-${ns}-${ps}.log
        done
    done
}

exec ./target/congress_network/congress.edgelist directed 5 3 10000 10 0.125 twitter
exec ./target/facebook_combined.txt.gz undirected 5 3 10000 10 0.125 facebook
