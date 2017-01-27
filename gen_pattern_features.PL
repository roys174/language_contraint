#!/usr/bin/env perl -w

use strict;
use MyUtils;

my %CW_TAGS = ("N" => undef, "V" => undef, "J" => undef, "R" => undef);

sub main(@) {
	my $min_thr = 5;
	my $max_patt_len = 5;
	my $if;
	my $patt_file;
	my $of;
	my $use_len;
	my $char_ngrams=4;
	
	MyUtils::usage({
		"if=s" => \$if,
		"patt_file=s" 					=> \$patt_file,
		"of=s"							=> \$of,
		"max_patt_len=i" 				=> \$max_patt_len,
		"min_thr=i"						=> \$min_thr,
		"char_ngrams=i"					=> \$char_ngrams,
		"len+"							=> \$use_len,
	}, ["if=s","of=s","patt_file=s"]);
	
	$max_patt_len--;
	my %saved;

	my $max_len = 0;
	
	if (-e $patt_file) {
		my $ifh = openRFile($patt_file);
		
		$max_len = $ifh->getline();
		chomp($max_len);
		while (my $line = $ifh->getline()) {
			chomp($line);
			my ($patt, $n, $sum, $max) = split("	", $line);
			$saved{$patt} = [$n, $max];
		}
		
		$ifh->close();
	} else {	
		my %patts;
		my $ifh = openRFile($if);

		my $is_first = 1;
		while (my $line = $ifh->getline()) {
			chomp($line);
	
			my ($label, $text) = split("\t", $line);
	
			my @w = split(" ", $text);
			
			if (defined $use_len and @w > $max_len) {
				$max_len = @w;
			}

			my $patts = get_patts(\@w, $max_patt_len, $char_ngrams);

			foreach my $patt (keys %$patts) {
				unless (exists $patts{$patt}) {
					$patts{$patt} = [$patts->{$patt}, $patts->{$patt}];
				} else {
					$patts{$patt}->[0] += $patts->{$patt};
					if ($patts{$patt}->[1] < $patts->{$patt}) {
						$patts{$patt}->[1] = $patts->{$patt}
					}
				}
			}
			$is_first = 0;
		}
		$ifh->close();
		
		my $i = 0;
		my @sk = sort {$patts{$b}->[0] <=> $patts{$a}->[0]} keys %patts;

		my $ofh = openWFile($patt_file);
		
		$ofh->print($max_len."\n");
		foreach my $k (@sk) {
			last if $patts{$k}->[0] < $min_thr;
			$ofh->print("$k\t$i\t$patts{$k}->[0]\t$patts{$k}->[1]\n");
			$saved{$k} = [$i++, $patts{$k}->[1]];
		}
		$ofh->close();
	}

	my $n_feats = scalar(keys %saved)+defined($use_len);
	print $n_feats." found\n";
	
	unless ($n_feats) {
		return 1;
	}

	
	my $ifh = openRFile($if);
	my $ofh = openWFile($of);
	
	while (my $line = $ifh->getline()) {
		chomp($line);
		
		my ($label, $text) = split("\t", $line);
		
		my @w = split(" ", $text);

		my $patts = get_patts(\@w, $max_patt_len, $char_ngrams);
		
		my %ts;
		
		foreach my $patt (keys %$patts) {
			if (exists $saved{$patt}) {
				$ts{$saved{$patt}->[0]} = ($patts->{$patt})/($saved{$patt}->[1]);	
			}
		}
		
		if ($use_len) {
			$ts{$n_feats-1} = scalar(@w)/$max_len;
		}
		
		$ofh->print("$label\t$text\t");		
		
		$ofh->print(join(" ", map { $_.":".sprintf("%.3g",$ts{$_})} keys %ts));
		$ofh->print("\n");
	}
	
	$ofh->close();
	$ifh->close();
		
	return 0;
}

sub get_patts($$$) {
	my $w = shift;
	unshift(@$w, "START_UNK");
	my $max_patt_len = shift;
	my $char_ngrams = shift;
	
	my %patts;
	
	# print "@w\n";
	my @w = map { exists($CW_TAGS{substr((split("_", $_))[1],0,1)}) ? (split("_", $_))[1] : (split("_", $_))[0] } @$w;
	
#	print "@w\n";
	# die;
	
	foreach my $i (0 .. $max_patt_len) {
		foreach my $j (0 .. @w-$i-1) {
			# print "$j ".($j+$i)."\n";
			my $p = join(" ", @w[$j..$j+$i]);
			$patts{$p}++;
		}
	}
	
	if ($char_ngrams) {
		my @w2 = map { (split("_",$_))[0]}  @$w;
		my $l = join(" ", @w2);
	
		foreach my $i (0 .. length($l)-$char_ngrams) {
			$patts{substr($l,$i,$char_ngrams)}++;
		}
	}
	
	return \%patts;
}

exit(main(@ARGV));