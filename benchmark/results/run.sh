#!/bin/bash

# Define ANSI color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if the bin path is provided as an argument
if [ -z "$1" ]; then
    echo -e "${RED}Usage: $0 <bin_path>${NC}"
    exit 1
fi

# Assign the first argument to a variable
BIN_PATH=$1

# Define the queries to run
queries=(q1 q4 q6 q12 q15 q17 q19)

# Loop through each query
for query in "${queries[@]}"
do
    echo -e "${BLUE}Running query ${query} with --nocache option...${NC}"
    ${BIN_PATH}/tpch_${query} --rows 256 1024 4096 --nocache --output ${query}_nc.csv
    echo -e "${GREEN}Output saved to ${query}_nc.csv${NC}"

    echo -e "${BLUE}Running query ${query} with --nofastcomp option...${NC}"
    ${BIN_PATH}/tpch_${query} --rows 256 1024 4096 --nofastcomp --output ${query}_base.csv
    echo -e "${GREEN}Output saved to ${query}_base.csv${NC}"

    echo -e "${BLUE}Running base query ${query}...${NC}"
    ${BIN_PATH}/tpch_${query} --rows 256 1024 4096 --output ${query}.csv
    echo -e "${GREEN}Output saved to ${query}.csv${NC}"
done

echo -e "${GREEN}All queries have been executed.${NC}"
