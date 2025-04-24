#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <fstream>
#include <set>
#include <algorithm>


struct Rating
{
    int user;
    int item;
    float rating;
    int timestamp;
};

struct ExtendSample
{
    int user;
    int positiveItem;
    int negativeItem;
};

std::vector<Rating> loadRatings(const std::filesystem::path& filepath)
{
    std::vector<Rating> ratings;
    std::ifstream file(filepath);
    if (!file.is_open())
    {
        std::cerr << "Failed to open file: " << filepath << std::endl;
        return ratings;
    }

    std::string line;
    while (std::getline(file, line))
    {
        int user, item, timestamp;
        float rating;
        if (sscanf(line.c_str(), "%d\t%d\t%f\t%d", &user, &item, &rating, &timestamp) == 4)
        {
            ratings.push_back({user, item, rating, timestamp});
        }
    }
    file.close();
    return ratings;
}

std::vector<ExtendSample> negativeSampling(std::vector<Rating>& ratings, const int num_negatives)
{
    std::cout << "Starting negative sampling..." << std::endl;
    int maxItem = 0;
    std::set<std::tuple<int, int> > positiveSamples;
    for (const Rating& rating : ratings)
    {
        positiveSamples.insert(std::make_tuple(rating.user, rating.item));
        maxItem = maxItem > rating.item ? maxItem : rating.item;
    }
    std::vector<ExtendSample> extendSamples;
    int cnt = 0;
    int size = ratings.size();
    int refreshRate = std::max(size / 100, 1);
    for (const Rating& rating : ratings)
    {
        int user = rating.user;
        int positiveItem = rating.item;
        for (int i = 0; i < num_negatives; i ++)
        {
            int negativeItem = rand() % maxItem + 1;
            while (positiveSamples.find(std::make_tuple(user, negativeItem)) != positiveSamples.end())
            {
                negativeItem = rand() % maxItem + 1;
            }
            extendSamples.push_back({user, positiveItem, negativeItem});
        }
        // std::cout << "[" << cnt ++ << "/" << size << "]" << std::endl;
        
        cnt ++;
        if (cnt % refreshRate == 0 || cnt == size)
        {
            float progress = cnt * 1.0 / size * 100;
            std::cout << "\rProgress: [" << std::string((int)(progress / 2), '=') 
                    << ">" << std::string(50 - (int)(progress / 2), ' ') << "] "
                    << cnt << "/" << size << " (" << (int)progress << "%)" << std::flush;
        }
    }
    std::cout << std::endl;
    return extendSamples;
}

void saveExtendSamples(const std::filesystem::path& filepath, const std::vector<ExtendSample>& extendSamples)
{
    std::cout << "Saving negative samples..." << std::endl;
    std::ofstream file(filepath);
    if (!file.is_open())
    {
        std::cerr << "Faild to open file: " << filepath << std::endl;
        return ;
    }
    int cnt = 0;
    int size = extendSamples.size();
    int refreshRate = std::max(size / 100, 1);
    for (const ExtendSample& extendSample : extendSamples)
    {
        file << extendSample.user << "\t" << extendSample.positiveItem << "\t" << extendSample.negativeItem << std::endl;
        cnt ++;
        if (cnt % refreshRate == 0 || cnt == size)
        {
            float progress = cnt * 1.0 / size * 100;
            std::cout << "\rProgress: [" << std::string((int)(progress / 2), '=') 
                    << ">" << std::string(50 - (int)(progress / 2), ' ') << "] "
                    << cnt << "/" << size << " (" << (int)progress << "%)" << std::flush;
        }
    }
    file.close();
    std::cout << std::endl;
    std::cout << "Negative sampling completed." << std::endl;
}

int main(int argc, char* argv[])
{
    std::filesystem::path prefix = "data";
    std::string dataset = argv[1];  // ['amazon-music', 'ciao', 'douban-book', 'douban-movie', 'ml-1m', 'ml-10m']
    // std::string fold_index = argv[2];  // ['1', '2', '3', '4', '5']
    std::string suffix_load = ".train";
    std::string suffix_save = ".extend";

    std::filesystem::path load_filepath;
    std::filesystem::path save_filepath;
    if (dataset == "ciao")
    {
        load_filepath = "." / prefix / "Ciao" / ("movie-ratings" + suffix_load);
        save_filepath = "." / prefix / "Ciao" / ("movie-ratings" + suffix_save);
    }
    else if (dataset == "douban-book")
    {
        load_filepath = prefix / "Douban" / "book" / ("douban_book" + suffix_load);
        save_filepath = prefix / "Douban" / "book" / ("douban_book" + suffix_save);
    }
    else if (dataset == "douban-movie")
    {
        load_filepath = prefix / "Douban" / "movie" / ("douban_movie" + suffix_load);
        save_filepath = prefix / "Douban" / "movie" / ("douban_movie" + suffix_save);
    }
    else if (dataset == "ml-1m")
    {
        load_filepath = prefix / "ml-1m" / ("ratings" + suffix_load);
        save_filepath = prefix / "ml-1m" / ("ratings" + suffix_save);
    }
    else if (dataset == "example")
    {
        load_filepath = prefix / "RunningExample" / ("running_example" + suffix_load);
        save_filepath = prefix / "RunningExample" / ("running_example" + suffix_save);
    }
    std::cout << "Load path: " << load_filepath << std::endl;
    std::cout << "Save path: " << save_filepath << std::endl;
    std::vector<Rating> ratings = loadRatings(load_filepath);
    if (ratings.empty())
    {
        std::cerr << "No ratings data loaded." << std::endl;
        return 1;
    }
    std::vector<ExtendSample> extendSamples = negativeSampling(ratings, 4);
    if (extendSamples.empty())
    {
        std::cerr << "Generate ExtendSample error." << std::endl;
        return 1;
    }
    saveExtendSamples(save_filepath, extendSamples);
    return 0;
}