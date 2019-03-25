# ames data
ames <- AmesHousing::make_ames()
library(rsample)
library(recipes)
library(dplyr)

# split data
set.seed(8451)
split <- initial_split(ames, strata = "Sale_Price")
ames_train <- training(split)
ames_test <- testing(split)

blueprint <- recipe(Sale_Price ~ ., data = ames) %>%
  step_other(all_nominal())

trained_recipe <- prep(blueprint, training = ames)
train <- bake(trained_recipe, new_data = ames_train)
test <- bake(trained_recipe, new_data = ames_test)

lm1 <- lm(Sale_Price ~ ., data = train)
predict(lm1, test)

rf1 <- ranger::ranger(Sale_Price ~ ., data = train, num.trees = 800)

readr::write_csv(train, path = "docs/data/train.csv")
readr::write_csv(subset(test, select = -Sale_Price), path = "docs/data/test.csv")

solution <- dplyr::select(test, Sale_Price) %>%
  mutate(Id = row_number())

readr::write_csv(solution, path = "docs/data/solution.csv")

predictions <- predict(rf1, test)$predictions
sample_submission <- data.frame(Id = solution$Id, Sale_Price = predictions)
readr::write_csv(sample_submission, path = "docs/data/sample_submission.csv")
